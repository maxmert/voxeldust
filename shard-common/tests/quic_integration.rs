use std::time::Duration;

use glam::{DQuat, DVec3};
use tokio::time::timeout;

use voxeldust_core::handoff::{GhostUpdate, HandoffAccepted, PlayerHandoff};
use voxeldust_core::shard_message::ShardMsg;
use voxeldust_core::shard_types::{SessionToken, ShardId};
use voxeldust_shard_common::quic_transport::{test_endpoint_pair, QuicTransport};

fn make_handoff() -> PlayerHandoff {
    PlayerHandoff {
        session_token: SessionToken(42),
        player_name: "Cosmonaut".to_string(),
        position: DVec3::new(0.0, 150000.0, 0.0),
        velocity: DVec3::ZERO,
        rotation: DQuat::IDENTITY,
        forward: DVec3::NEG_Z,
        fly_mode: false,
        speed_tier: 0,
        grounded: true,
        health: 100.0,
        shield: 100.0,
        source_shard: ShardId(1),
        source_tick: 500,
        target_star_index: None,
        galaxy_context: None,
        target_planet_seed: None,
        target_planet_index: None,
        target_ship_id: None,
        target_ship_shard_id: None,
        ship_system_position: None,
        ship_rotation: None,
        game_time: 0.0,
    }
}

#[tokio::test]
async fn send_and_receive_handoff() {
    let (transport_a, transport_b) = test_endpoint_pair().await.unwrap();

    let peer_id = ShardId(2);
    let peer_addr = transport_b.local_addr();

    // Spawn receiver.
    let recv_handle = tokio::spawn({
        let transport_b = transport_b.clone();
        async move {
            let incoming = timeout(Duration::from_secs(5), transport_b.accept())
                .await
                .expect("accept timed out")
                .expect("no incoming connection");

            let msg = timeout(Duration::from_secs(5), incoming.recv())
                .await
                .expect("recv timed out")
                .expect("recv failed");

            msg
        }
    });

    // Send handoff.
    let msg = ShardMsg::PlayerHandoff(make_handoff());
    transport_a.send(peer_id, peer_addr, &msg).await.unwrap();

    // Verify received.
    let received = recv_handle.await.unwrap();
    if let ShardMsg::PlayerHandoff(h) = received {
        assert_eq!(h.session_token, SessionToken(42));
        assert_eq!(h.player_name, "Cosmonaut");
        assert!((h.position.y - 150000.0).abs() < 1e-10);
        assert!(h.grounded);
        assert_eq!(h.source_shard, ShardId(1));
    } else {
        panic!("expected PlayerHandoff, got {:?}", received);
    }

    transport_a.shutdown();
    transport_b.shutdown();
}

#[tokio::test]
async fn send_multiple_messages_on_same_connection() {
    let (transport_a, transport_b) = test_endpoint_pair().await.unwrap();

    let peer_id = ShardId(3);
    let peer_addr = transport_b.local_addr();

    let recv_handle = tokio::spawn({
        let transport_b = transport_b.clone();
        async move {
            let incoming = timeout(Duration::from_secs(5), transport_b.accept())
                .await
                .expect("accept timed out")
                .expect("no incoming");

            let mut messages = Vec::new();
            for _ in 0..3 {
                let msg = timeout(Duration::from_secs(5), incoming.recv())
                    .await
                    .expect("recv timed out")
                    .expect("recv failed");
                messages.push(msg);
            }
            messages
        }
    });

    // Send 3 different message types.
    let messages = vec![
        ShardMsg::PlayerHandoff(make_handoff()),
        ShardMsg::HandoffAccepted(HandoffAccepted {
            session_token: SessionToken(42),
            target_shard: ShardId(3),
        }),
        ShardMsg::GhostUpdate(GhostUpdate {
            session_token: SessionToken(42),
            position: DVec3::new(1.0, 2.0, 3.0),
            rotation: DQuat::IDENTITY,
            velocity: DVec3::ZERO,
            tick: 501,
        }),
    ];

    for msg in &messages {
        transport_a.send(peer_id, peer_addr, msg).await.unwrap();
    }

    let received = recv_handle.await.unwrap();
    assert_eq!(received.len(), 3);
    assert!(matches!(received[0], ShardMsg::PlayerHandoff(_)));
    assert!(matches!(received[1], ShardMsg::HandoffAccepted(_)));
    assert!(matches!(received[2], ShardMsg::GhostUpdate(_)));

    transport_a.shutdown();
    transport_b.shutdown();
}

#[tokio::test]
async fn circuit_breaker_opens_on_unreachable_peer() {
    let transport = QuicTransport::bind("127.0.0.1:0".parse().unwrap())
        .await
        .unwrap();

    let fake_peer = ShardId(999);
    // Connect to a port that nothing is listening on.
    let unreachable_addr = "127.0.0.1:1".parse().unwrap();

    let msg = ShardMsg::Heartbeat(voxeldust_core::shard_types::ShardHeartbeat {
        shard_id: ShardId(1),
        tick_ms: 50.0,
        p99_tick_ms: 50.0,
        player_count: 0,
        chunk_count: 0,
    });

    // Send 5 times — should fail each time (connect timeout or connection refused).
    for i in 0..5 {
        let result = transport.send(fake_peer, unreachable_addr, &msg).await;
        assert!(result.is_err(), "attempt {i} should fail");
    }

    // 6th attempt should be rejected by circuit breaker immediately.
    let result = transport.send(fake_peer, unreachable_addr, &msg).await;
    assert!(
        matches!(
            result,
            Err(voxeldust_shard_common::quic_transport::TransportError::CircuitBreakerOpen(_))
        ),
        "expected CircuitBreakerOpen, got {:?}",
        result
    );

    transport.shutdown();
}
