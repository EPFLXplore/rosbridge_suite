# Software License Agreement (BSD License)
#
# Copyright (c) 2012, Willow Garage, Inc.
# Copyright (c) 2013, PAL Robotics SL
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

from functools import partial
from threading import Lock, RLock
from typing import TYPE_CHECKING, Generic, cast

from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from rosbridge_library.internal import ros_loader
from rosbridge_library.internal.message_conversion import msg_class_type_repr
from rosbridge_library.internal.outgoing_message import OutgoingMessage
from rosbridge_library.internal.topics import (
    TopicNotEstablishedException,
    TypeConflictException,
)
from rosbridge_library.internal.type_support import ROSMessageT

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from rclpy.node import Node
    from rclpy.subscription import Subscription


""" Manages and interfaces with ROS Subscriber objects.  A single subscriber
is shared between multiple clients
"""

# How often default-QoS subscriptions re-check the publishers on their topic. 5s is well inside
# the time it takes an operator to notice a stack they just started is missing from the UI.
#
# One timer on the SubscriberManager drives every MultiSubscriber, rather than one timer each:
# get_publishers_info_by_topic takes the rcl graph lock, and that is the same lock service calls
# need to discover their server, so N independent timers turn into a steady drip of contention
# against exactly the path an operator is waiting on.
QOS_RENEGOTIATE_PERIOD_S = 5.0


class MultiSubscriber(Generic[ROSMessageT]):
    """
    Handles multiple clients for a single subscriber.

    Converts msgs to JSON before handing them to callbacks. Due to subscriber
    callbacks being called in separate threads, must lock whenever modifying
    or accessing the subscribed clients.
    """

    def __init__(
        self,
        topic: str,
        client_id: str,
        callback: Callable[[OutgoingMessage[ROSMessageT]], None],
        node_handle: Node,
        msg_type: str | None = None,
        raw: bool = False,
        qos: QoSProfile | None = None,
    ) -> None:
        """
        Register a subscriber on the specified topic.

        :param topic: The name of the topic to register the subscriber on
        :param client_id The ID of the client subscribing
        :param callback: This client's callback, that will be called for incoming messages
        :param node_handle: Handle to a rclpy node to create the publisher
        :param msg_type: (optional) The type to register the subscriber as.  If not provided, an
            attempt will be made to infer the topic type
        :param qos: (optional) The QoS profile to register the subscriber with. If not provided,
            mirrors the publishers on the topic with KEEP_LAST depth 1 (see
            _get_default_qos_profile).

        :raises TopicNotEstablishedException: If no msg_type was specified by the caller and the
            topic is not yet established, so a topic type cannot be inferred
        :raises TypeConflictException: If the msg_type was specified by the caller and the topic
            is established, and the established type is different to the user-specified msg_type
        """
        # First check to see if the topic is already established
        topics_names_and_types = dict(node_handle.get_topic_names_and_types())
        topic_types = topics_names_and_types.get(topic)

        # If it's not established and no type was specified, exception
        if msg_type is None and topic_types is None:
            raise TopicNotEstablishedException(topic)

        # topic_types is a list of types or None at this point; only one type is supported.
        topic_type: str | None = None
        if topic_types is not None:
            if len(topic_types) > 1:
                node_handle.get_logger().warning(
                    f"More than one topic type detected: {topic_types}"
                )
            topic_type = topic_types[0]

        # Use the established topic type if none was specified
        if msg_type is None:
            assert topic_type is not None
            msg_type = topic_type

        # Load the message class, propagating any exceptions from bad msg types
        msg_class = cast("type[ROSMessageT]", ros_loader.get_message_class(msg_type))

        # Make sure the specified msg type and established msg type are same
        msg_type_string = msg_class_type_repr(msg_class)
        if topic_type is not None and topic_type != msg_type_string:
            raise TypeConflictException(topic, topic_type, msg_type_string)

        # An explicit profile is the client's choice and is never renegotiated below.
        self.qos_is_explicit = qos is not None

        if qos is None:
            qos = self._get_default_qos_profile(node_handle, topic)

        # Create the subscriber and associated member variables
        # Subscriptions is initialized with the current client to start with.
        self.subscriptions = {client_id: callback}
        self.rlock = RLock()
        self.msg_class = msg_class
        self.node_handle = node_handle
        self.topic = topic
        self.qos_profile: QoSProfile = qos
        self.raw = raw
        self.callback_group = MutuallyExclusiveCallbackGroup()

        self.subscriber = node_handle.create_subscription(
            msg_class,
            topic,
            partial(self.callback, callbacks=None),
            qos_profile=self.qos_profile,
            raw=raw,
            callback_group=self.callback_group,
        )
        self.new_subscriber: Subscription | None = None
        self.new_subscriptions: dict[str, Callable[[OutgoingMessage[ROSMessageT]], None]] = {}

        # Renegotiation is driven by SubscriberManager's single shared timer, which skips
        # subscribers whose profile the client chose explicitly.
        self.unregistered = False

    def _get_default_qos_profile(self, node_handle: Node, topic: str) -> QoSProfile:
        """
        Default QoS when the rosbridge client omits qos on subscribe.

        Mirrors the publishers currently on the topic, with KEEP_LAST depth 1 so a slow websocket
        sink cannot build a DDS-side backlog. Matching matters over a lossy link: a BEST_EFFORT
        reader gets no retransmission, so a sample split across several UDP datagrams is lost
        entirely if one of them is dropped, and a VOLATILE reader never receives the last value
        held by a TRANSIENT_LOCAL publisher. `ros2 topic echo` does the same introspection, which
        is why it can show a topic the control station reports as having no data.

        A topic with no publisher yet cannot be introspected, so it starts on the weak fallback
        below. `renegotiate_qos` re-runs this once publishers show up and rebuilds the reader if
        the answer changed, which is what lets a stack started after the browser connected be
        picked up without the client resubscribing.

        Clients may still pass an explicit qos object in the subscribe message to override this
        (the camera feeds do, to stay BEST_EFFORT regardless of how the camera node publishes);
        an explicit profile is never renegotiated.
        """
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        infos = node_handle.get_publishers_info_by_topic(topic)
        if not infos:
            return qos

        # Only upgrade when *every* publisher offers the stronger policy: a BEST_EFFORT writer
        # does not match a RELIABLE reader at all, so a single one would silence the whole
        # subscription. Same reasoning for VOLATILE vs TRANSIENT_LOCAL.
        if all(pub.qos_profile.reliability == ReliabilityPolicy.RELIABLE for pub in infos):
            qos.reliability = ReliabilityPolicy.RELIABLE
        if all(pub.qos_profile.durability == DurabilityPolicy.TRANSIENT_LOCAL for pub in infos):
            qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

        return qos

    def renegotiate_qos(self) -> None:
        """
        Re-match the subscription QoS against the publishers currently on the topic.

        The profile is chosen when the reader is created, which is wrong as soon as the graph
        changes: a topic subscribed while its stack was down is pinned to the BEST_EFFORT/VOLATILE
        fallback for the life of the process, and over a lossy link that means no retransmission
        and lost samples long after the stack came back. Reloading the browser used to be the only
        cure, because a fresh subscribe re-ran the introspection.

        Two rules keep this from becoming the resubscribe churn it replaces:
        - nothing publishing the topic means no information to act on, so leave the reader alone
          rather than downgrading it, which is what makes a stopped stack completely quiet here;
        - rebuild only when the desired profile actually differs, so the steady state costs one
          graph query every QOS_RENEGOTIATE_PERIOD_S and nothing else.

        Downgrades are applied as readily as upgrades. A reader left on RELIABLE after a node
        restarts as BEST_EFFORT does not match the writer at all, so the goal is to equal the
        desired profile, not to strengthen it.
        """
        if self.qos_is_explicit or self.unregistered:
            return

        infos = self.node_handle.get_publishers_info_by_topic(self.topic)
        if not infos:
            return

        desired = self._get_default_qos_profile(self.node_handle, self.topic)

        with self.rlock:
            current = self.qos_profile
            if (
                desired.reliability == current.reliability
                and desired.durability == current.durability
            ):
                return

            self.node_handle.get_logger().info(
                f"QoS for {self.topic} renegotiated: "
                f"{current.reliability.name}/{current.durability.name} -> "
                f"{desired.reliability.name}/{desired.durability.name}"
            )

            self.qos_profile = desired

            old_subscriber = self.subscriber
            self.subscriber = self.node_handle.create_subscription(
                self.msg_class,
                self.topic,
                partial(self.callback, callbacks=None),
                qos_profile=self.qos_profile,
                raw=self.raw,
                callback_group=self.callback_group,
            )
            self._schedule_destroy_subscription(old_subscriber)

            # A new_subscriber is live only while a client is waiting for its first message; it
            # has to move to the new profile too or that client keeps waiting on the old reader.
            if self.new_subscriber is not None:
                old_new_subscriber = self.new_subscriber
                self.new_subscriber = self.node_handle.create_subscription(
                    self.msg_class,
                    self.topic,
                    self._new_sub_callback,
                    qos_profile=self.qos_profile,
                    raw=self.raw,
                    callback_group=self.callback_group,
                )
                self._schedule_destroy_subscription(old_new_subscriber)

    def _schedule_destroy_subscription(self, subscription: Subscription[ROSMessageT]) -> None:
        """
        Schedule subscription destruction on the executor thread.

        Used to avoid race conditions between executor and non-executor threads.

        Args:
            subscription (Subscription[ROSMessageT]): Subscription to destroy

        """
        executor = self.node_handle.executor
        if executor is not None:
            executor.create_task(self.node_handle.destroy_subscription, subscription)
        else:
            self.node_handle.destroy_subscription(subscription)

    def unregister(self) -> None:
        self.unregistered = True
        self._schedule_destroy_subscription(self.subscriber)
        with self.rlock:
            self.subscriptions.clear()
            if self.new_subscriber:
                self._schedule_destroy_subscription(self.new_subscriber)
                self.new_subscriber = None

    def verify_type(self, msg_type: str) -> None:
        """
        Verify that the subscriber subscribes to messages of this type.

        :param msg_type: The type to check this subscriber against

        :raises Exception: If ros_loader cannot load the specified msg type
        :raises TypeConflictException: If the msg_type is different than the type of this publisher
        """
        if ros_loader.get_message_class(msg_type) is not self.msg_class:
            raise TypeConflictException(self.topic, msg_class_type_repr(self.msg_class), msg_type)

    def subscribe(
        self, client_id: str, callback: Callable[[OutgoingMessage[ROSMessageT]], None]
    ) -> None:
        """
        Subscribe the specified client to this subscriber.

        :param client_id: The ID of the client subscribing
        :param callback: This client's callback, that will be called for incoming messages
        """
        with self.rlock:
            # If the topic is latched, adding a new subscriber will immediately invoke
            # the given callback.
            # In any case, the first message is handled using new_sub_callback,
            # which adds the new callback to the subscriptions dictionary.
            self.new_subscriptions.update({client_id: callback})

            if self.new_subscriber is None:
                self.new_subscriber = self.node_handle.create_subscription(
                    self.msg_class,
                    self.topic,
                    self._new_sub_callback,
                    qos_profile=self.qos_profile,
                    raw=self.raw,
                    callback_group=self.callback_group,
                )

    def unsubscribe(self, client_id: str) -> None:
        """
        Unsubscribe the specified client from this subscriber.

        :param client_id: The ID of the client to unsubscribe
        """
        with self.rlock:
            if client_id in self.new_subscriptions:
                del self.new_subscriptions[client_id]
            if client_id in self.subscriptions:
                del self.subscriptions[client_id]

    def has_subscribers(self) -> bool:
        """Return true if there are subscribers."""
        with self.rlock:
            return len(self.subscriptions) + len(self.new_subscriptions) != 0

    def callback(
        self, msg: ROSMessageT, callbacks: Iterable[Callable[[OutgoingMessage], None]] | None = None
    ) -> None:
        """
        Handle incoming messages on the rclpy subscription.

        Passes the message to registered subscriber callbacks.

        :param msg: The ROS message coming from the subscriber
        :param callbacks: Subscriber callbacks to invoke
        """
        outgoing = OutgoingMessage(msg)

        with self.rlock:
            callbacks = callbacks or self.subscriptions.values()

            # Pass the JSON to each of the callbacks
            for callback in callbacks:
                try:
                    callback(outgoing)
                except Exception as exc:  # noqa: PERF203
                    # Do nothing if one particular callback fails except log it
                    self.node_handle.get_logger().error(
                        f"Exception calling subscribe callback: {exc}"
                    )

    def _new_sub_callback(self, msg: ROSMessageT) -> None:
        """
        Callbacks for new subscribers.

        If the topic was latched, a new subscriber has to be added to receive
        a new message and route it to the new subscriptor.

        After the first message is routed, the new subscriber is deleted and
        the subscriptions dictionary is updated with the newly incorporated
        subscriptors.
        """
        with self.rlock:
            # return if work has already been done by another callback invocation
            if self.new_subscriber is None:
                if self.new_subscriptions:
                    self.node_handle.get_logger().error(
                        "new_subscriber is None but new_subscriptions is not empty; "
                        "this should never happen"
                    )
                return
            self.callback(msg, list(self.new_subscriptions.values()))
            self.subscriptions.update(self.new_subscriptions)
            self.new_subscriptions = {}
            self._schedule_destroy_subscription(self.new_subscriber)
            self.new_subscriber = None


class SubscriberManager:
    """Keeps track of client subscriptions."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._subscribers: dict[str, MultiSubscriber] = {}
        self._qos_timer = None
        self._qos_timer_node: Node | None = None

    def _ensure_qos_timer(self, node_handle: Node) -> None:
        """
        Start the shared QoS renegotiation timer on first use.

        Created lazily rather than in __init__ because `manager` is a module-level singleton
        built at import time, long before there is a node to hang a timer off. For the same
        reason the owning node is tracked: the singleton outlives any individual node, so a
        timer left over from a previous one has to be replaced rather than reused.
        """
        if self._qos_timer is not None and self._qos_timer_node is node_handle:
            return

        self._qos_timer = node_handle.create_timer(QOS_RENEGOTIATE_PERIOD_S, self._renegotiate_qos)
        self._qos_timer_node = node_handle

    def _renegotiate_qos(self) -> None:
        """Re-match every default-QoS subscription against the publishers on its topic."""
        with self._lock:
            subscribers = list(self._subscribers.values())

        # Deliberately outside the lock: renegotiating rebuilds subscriptions, and holding the
        # manager lock across that would block every subscribe/unsubscribe for the duration.
        for subscriber in subscribers:
            subscriber.renegotiate_qos()

    def subscribe(
        self,
        client_id: str,
        topic: str,
        callback: Callable[[OutgoingMessage], None],
        node_handle: Node,
        msg_type: str | None = None,
        raw: bool = False,
        qos: QoSProfile | None = None,
    ) -> None:
        """
        Subscribe to a topic.

        Subscribers are shared between clients, so a single MultiSubscriber
        instance is created per topic, even if multiple clients subscribe to the same topic.
        The QoS profile is determined at the first registration of a subscriber.

        :param client_id: The ID of the client making this subscribe request
        :param topic: The name of the topic to subscribe to
        :param callback: The callback to call for incoming messages on the topic
        :param msg_type: (optional) The type of the topic
        :param qos: (optional) The QoSProfile of the topic
        """
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = MultiSubscriber(
                    topic, client_id, callback, node_handle, msg_type=msg_type, raw=raw, qos=qos
                )
            else:
                self._subscribers[topic].subscribe(client_id, callback)

            if msg_type is not None and not raw:
                self._subscribers[topic].verify_type(msg_type)

            self._ensure_qos_timer(node_handle)

    def unsubscribe(self, client_id: str, topic: str) -> None:
        """
        Unsubscribe from a topic.

        :param client_id: The ID of the client to unsubscribe
        :param topic: The topic to unsubscribe from
        """
        with self._lock:
            if topic not in self._subscribers:
                return

            self._subscribers[topic].unsubscribe(client_id)

            if not self._subscribers[topic].has_subscribers():
                self._subscribers[topic].unregister()
                del self._subscribers[topic]


manager = SubscriberManager()
