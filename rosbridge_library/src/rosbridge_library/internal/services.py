# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Willow Garage, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of Willow Garage, Inc. nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, Any

from rclpy.callback_groups import ReentrantCallbackGroup

from rosbridge_library.internal.message_conversion import (
    extract_values,
    populate_instance,
)
from rosbridge_library.internal.ros_loader import (
    get_service_class,
    get_service_request_instance,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from rclpy.client import Client
    from rclpy.node import Node

    from rosbridge_library.internal.type_support import ROSMessage


class InvalidServiceException(Exception):
    def __init__(self, service_name: str) -> None:
        Exception.__init__(self, f"Service {service_name} does not exist")


# Service clients are cached and reused across calls, keyed by resolved service name.
#
# Creating one per call is very expensive on a real robot network: it costs a
# get_service_names_and_types() sweep of the whole graph (hundreds of entries once a navigation
# stack is up, taken under the rcl graph lock) plus a full SEDP endpoint discovery round-trip in
# wait_for_service(). That is paid on every operator button press, and when discovery does not
# finish inside server_ready_timeout the call fails with InvalidServiceException even though the
# server is perfectly healthy.
#
# The cache lives on the node rather than in a module global so its lifetime is exactly the
# node's: a destroyed node takes its clients with it and cannot hand a stale client to a later
# node that happens to reuse the same name (which is what tests do between cases).
_CLIENT_CACHE_ATTR = "_rosbridge_service_clients"
_client_cache_lock = Lock()


def _client_cache(node_handle: Node) -> dict[str, tuple[str, Client]]:
    """Return this node's {resolved service name: (service type, client)} cache."""
    cache = getattr(node_handle, _CLIENT_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(node_handle, _CLIENT_CACHE_ATTR, cache)
    return cache


def _discard_client(node_handle: Node, service: str, client: Client) -> None:
    """
    Drop `client` from the cache and destroy it, so the next call rediscovers.

    Only removes the entry if it is still the one passed in: another thread may already have
    replaced it after noticing the same failure, and destroying that fresh client instead would
    just move the problem to the next call.
    """
    with _client_cache_lock:
        cache = _client_cache(node_handle)
        if cache.get(service, (None, None))[1] is client:
            del cache[service]
    node_handle.destroy_client(client)


def _acquire_client(
    node_handle: Node, service: str, server_ready_timeout: float
) -> tuple[str, Client]:
    """
    Return (service type, client) for an already-resolved service name.

    :raises InvalidServiceException: If the service is not in the graph, or no server showed up
        within server_ready_timeout.
    """
    with _client_cache_lock:
        cached = _client_cache(node_handle).get(service)

    if cached is not None:
        # service_is_ready() is a local rcl check, not a graph query, so this is cheap.
        if cached[1].service_is_ready():
            return cached
        # Not ready is usually a blip -- a brief dropout on the rover link, or a server still
        # coming back up. Give the existing client the same grace a fresh one would get before
        # tearing it down: another thread may be mid-call on it, and destroying it under them
        # strands that call until its own timeout.
        if cached[1].wait_for_service(server_ready_timeout):
            return cached
        _discard_client(node_handle, service, cached[1])

    service_names_and_types = dict(node_handle.get_service_names_and_types())
    service_types = service_names_and_types.get(service)
    if service_types is None:
        raise InvalidServiceException(service)

    # service_type is a tuple of types at this point; only one type is supported.
    if len(service_types) > 1:
        node_handle.get_logger().warning(f"More than one service type detected: {service_types}")
    service_type = service_types[0]

    client: Client = node_handle.create_client(
        get_service_class(service_type),  # type: ignore[misc]  # Silence type checker about not being able to infer type
        service,
        callback_group=ReentrantCallbackGroup(),
    )

    if not client.wait_for_service(server_ready_timeout):
        node_handle.destroy_client(client)
        raise InvalidServiceException(service)

    with _client_cache_lock:
        cache = _client_cache(node_handle)
        existing = cache.get(service)
        if existing is not None:
            # Another thread discovered the same service concurrently. Keep its client so there
            # is only ever one per service, and drop the duplicate we just built.
            node_handle.destroy_client(client)
            return existing
        cache[service] = (service_type, client)

    return service_type, client


def dispose_service_clients(node_handle: Node) -> None:
    """Destroy every cached service client for this node. Intended for shutdown and tests."""
    with _client_cache_lock:
        cache = _client_cache(node_handle)
        clients = [client for _, client in cache.values()]
        cache.clear()
    for client in clients:
        node_handle.destroy_client(client)


class ServiceCaller(Thread):
    def __init__(
        self,
        service: str,
        args: list | dict[str, Any] | None,
        timeout: float,
        success_callback: Callable[[dict], None],
        error_callback: Callable[[Exception], None],
        node_handle: Node,
    ) -> None:
        """
        Create a service caller for the specified service.

        Use start() to start in a separate thread or run() to run in this thread.

        :param service: The name of the service to call
        :param args: Arguments to pass to the service.  Can be an ordered list, or a dict of
            name-value pairs. Anything else will be treated as though no arguments were provided
            (which is still valid for some kinds of service)
        :param timeout: The time, in seconds, to wait for a response from the server.
            A non-positive value means no timeout.
        :param success_callback: A callback to call with the JSON result of the service call
        :param error_callback: A callback to call if an error occurs. The callback will be passed
            the exception that caused the failure
        :param node_handle: A ROS 2 node handle to call services
        """
        Thread.__init__(self)
        self.daemon = True
        self.service = service
        self.args = args
        self.timeout = timeout
        self.success = success_callback
        self.error = error_callback
        self.node_handle = node_handle

    def run(self) -> None:
        try:
            # Call the service and pass the result to the success handler
            self.success(
                call_service(
                    self.node_handle,
                    self.service,
                    args=self.args,
                    server_response_timeout=self.timeout,
                )
            )
        except Exception as e:
            # On error, just pass the exception to the error handler
            self.error(e)


def args_to_service_request_instance(inst: ROSMessage, args: list | dict[str, Any] | None) -> None:
    """
    Populate a service request instance with the provided args.

    Propagates any exceptions that may be raised.

    :param args: Can be a dictionary of values, or a list, or None
    """
    msg = {}
    if isinstance(args, list):
        msg = dict(zip(inst.get_fields_and_field_types().keys(), args, strict=False))
    elif isinstance(args, dict):
        msg = args

    # Populate the provided instance, propagating any exceptions
    populate_instance(msg, inst)


def call_service(
    node_handle: Node,
    service: str,
    args: list | dict[str, Any] | None = None,
    server_ready_timeout: float = 1.0,
    server_response_timeout: float = 5.0,
) -> dict:
    # Get the fully qualified service name with remappings applied
    service = node_handle.resolve_service_name(service)

    # Given the service name, fetch the type and a client. Both come from the per-node cache, so
    # a repeated call to the same service costs neither a graph sweep nor endpoint discovery.
    service_type, client = _acquire_client(node_handle, service, server_ready_timeout)

    inst = get_service_request_instance(service_type)

    # Populate the instance with the provided args
    args_to_service_request_instance(inst, args)

    future = client.call_async(inst)
    event = Event()

    def future_done_callback() -> None:
        event.set()

    future.add_done_callback(lambda _: future_done_callback())

    if not event.wait(timeout=(server_response_timeout if server_response_timeout > 0 else None)):
        future.cancel()
        # A call that went unanswered says nothing good about this client, and keeping it cached
        # would make every later call to the same service wait out the timeout too.
        _discard_client(node_handle, service, client)
        msg = "Timeout exceeded while waiting for service response"
        raise Exception(msg)

    result = future.result()

    if result is not None:
        # Turn the response into JSON and pass to the callback
        json_response = extract_values(result)
    else:
        exception = future.exception()
        _discard_client(node_handle, service, client)
        raise Exception("Service call exception: " + str(exception))

    return json_response
