//! Stream and datagram framing (`docs/design/connection_plane.md` §1.2).
//!
//! Reliable streams carry length-delimited frames:
//! ```text
//! StreamFrame := u32_be total_len | u8 codec_flags | payload[total_len - 1]
//! ```
//! `codec_flags` bit0 selects the serializer — ALWAYS 0 (postcard) in v1; the bit is
//! reserved so bitcode can be added later without a wire break. bit1 (lz4, bulk only)
//! and the rest are likewise reserved-zero. Any set reserved bit is a hard error.
//!
//! Datagrams are QUIC-delimited; their headers are typed structs in `channels`.

use serde::{Serialize, de::DeserializeOwned};

/// Maximum stream frame payload. Matches the io-prod transport cap; both migrate
/// into `TransportTuning` when it lands (P1).
pub const MAX_STREAM_FRAME_BYTES: u32 = 1 << 20;

/// v1: postcard, uncompressed.
pub const CODEC_FLAGS_V1: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FrameError {
    #[error("frame truncated: have {have} bytes, need {need}")]
    Truncated { have: usize, need: usize },
    #[error("frame length {0} exceeds the {MAX_STREAM_FRAME_BYTES}-byte cap")]
    TooLarge(u64),
    #[error("reserved codec flags set: {0:#04x}")]
    ReservedCodecFlags(u8),
    #[error("payload failed to decode")]
    Decode,
    #[error("payload failed to encode")]
    Encode,
}

/// Frame an already-serialized payload. Monomorphic: all branching lives here
/// (the generic `encode_frame` below is a branchless shim — see CLAUDE.md HR5 note).
pub fn frame_payload(payload: Result<Vec<u8>, postcard::Error>) -> Result<Vec<u8>, FrameError> {
    let payload = payload.map_err(|_| FrameError::Encode)?;
    // Cap-check in usize space FIRST, so the u32 cast below is bounded (no
    // untestable 4 GiB try_from branch).
    if payload.len() + 1 > MAX_STREAM_FRAME_BYTES as usize {
        return Err(FrameError::TooLarge((payload.len() + 1) as u64));
    }
    #[allow(clippy::cast_possible_truncation)] // bounded by the cap check above
    let total_len = (payload.len() + 1) as u32;
    let mut out = Vec::with_capacity(4 + 1 + payload.len());
    out.extend_from_slice(&total_len.to_be_bytes());
    out.push(CODEC_FLAGS_V1);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Encode a message as a stream frame.
pub fn encode_frame<T: Serialize>(msg: &T) -> Result<Vec<u8>, FrameError> {
    frame_payload(postcard::to_allocvec(msg))
}

/// Split one frame off the front of `buf`. Monomorphic worker: returns the payload
/// slice and the total bytes consumed; `decode_frame` is the generic shim.
pub fn split_frame(buf: &[u8]) -> Result<(&[u8], usize), FrameError> {
    if buf.len() < 4 {
        return Err(FrameError::Truncated {
            have: buf.len(),
            need: 4,
        });
    }
    let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]);
    if total_len > MAX_STREAM_FRAME_BYTES {
        return Err(FrameError::TooLarge(u64::from(total_len)));
    }
    if total_len == 0 {
        return Err(FrameError::Truncated { have: 0, need: 1 });
    }
    let total = total_len as usize;
    if buf.len() < 4 + total {
        return Err(FrameError::Truncated {
            have: buf.len(),
            need: 4 + total,
        });
    }
    let codec_flags = buf[4];
    if codec_flags != CODEC_FLAGS_V1 {
        return Err(FrameError::ReservedCodecFlags(codec_flags));
    }
    Ok((&buf[5..4 + total], 4 + total))
}

/// Validate + strip the `codec_flags` byte off a stream-frame BODY — the `total_len` bytes that
/// follow the 4-byte length prefix a STREAMING reader has already consumed off the wire. A QUIC
/// reader reads length-then-body, so it cannot use [`split_frame`] (which works on a buffer that
/// still carries the prefix); this is the streaming counterpart, with the SAME reserved-bit
/// hard-reject — so io-prod's stream transports route through this ONE codec-flags home instead of
/// hand-rolling the discipline (HR3). The cap is enforced by the reader on `total_len` before it
/// allocates the body.
pub fn frame_body_payload(body: &[u8]) -> Result<&[u8], FrameError> {
    let &codec_flags = body
        .first()
        .ok_or(FrameError::Truncated { have: 0, need: 1 })?;
    if codec_flags != CODEC_FLAGS_V1 {
        return Err(FrameError::ReservedCodecFlags(codec_flags));
    }
    Ok(&body[1..])
}

/// Decode one frame's payload as `T`.
pub fn decode_frame<T: DeserializeOwned>(buf: &[u8]) -> Result<(T, usize), FrameError> {
    split_frame(buf).and_then(decode_split())
}

/// Monomorphic-closure factory keeping `decode_frame` branchless per instantiation.
fn decode_split<T: DeserializeOwned>() -> impl Fn((&[u8], usize)) -> Result<(T, usize), FrameError>
{
    |(payload, consumed)| {
        postcard::from_bytes(payload)
            .map(|msg| (msg, consumed))
            .map_err(|_| FrameError::Decode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn roundtrip_a_message() {
        let frame = encode_frame(&("hello", 42u32)).expect("encode");
        let (back, consumed): ((String, u32), usize) = decode_frame(&frame).expect("decode");
        assert_eq!(back, ("hello".to_string(), 42));
        assert_eq!(consumed, frame.len());
    }

    #[test]
    fn frames_concatenate_and_split_in_order() {
        let mut stream = encode_frame(&1u32).expect("a");
        stream.extend(encode_frame(&2u32).expect("b"));
        let (first, used): (u32, usize) = decode_frame(&stream).expect("first");
        let (second, used2): (u32, usize) = decode_frame(&stream[used..]).expect("second");
        assert_eq!((first, second), (1, 2));
        assert_eq!(used + used2, stream.len());
    }

    #[test]
    fn truncations_error_cleanly() {
        let frame = encode_frame(&[7u8; 16]).expect("encode");
        // [u8; 16] postcard-encodes to 16 bytes; total_len = 17; frame = 4 + 17.
        assert_eq!(frame.len(), 21);
        for cut in 0..frame.len() {
            let err = split_frame(&frame[..cut]).expect_err("truncated");
            let expected = if cut < 4 {
                FrameError::Truncated { have: cut, need: 4 }
            } else {
                FrameError::Truncated {
                    have: cut,
                    need: 21,
                }
            };
            assert_eq!(err, expected, "cut={cut}");
        }
    }

    #[test]
    fn reserved_codec_flags_are_rejected() {
        let mut frame = encode_frame(&0u8).expect("encode");
        frame[4] = 0b0000_0001; // claim bitcode: reserved in v1
        assert_eq!(
            split_frame(&frame).expect_err("flags"),
            FrameError::ReservedCodecFlags(1)
        );
    }

    #[test]
    fn frame_body_payload_strips_flags_and_rejects_reserved() {
        // The body is the `total_len` bytes a streaming reader reads AFTER the 4-byte prefix:
        // [codec_flags][payload]. `frame_body_payload` is `split_frame`'s streaming counterpart.
        let payload = postcard::to_allocvec(&("x", 9u32)).expect("postcard");
        let frame = frame_payload(Ok(payload.clone())).expect("frame");
        let body = &frame[4..];
        assert_eq!(frame_body_payload(body).expect("ok"), payload.as_slice());
        // An empty body (zero-length frame) and a reserved codec flag are both hard errors.
        assert_eq!(
            frame_body_payload(&[]).expect_err("empty"),
            FrameError::Truncated { have: 0, need: 1 }
        );
        let mut bad = body.to_vec();
        bad[0] = 0b0000_0001; // claim bitcode: reserved in v1
        assert_eq!(
            frame_body_payload(&bad).expect_err("flag"),
            FrameError::ReservedCodecFlags(1)
        );
    }

    #[test]
    fn zero_length_frame_is_invalid() {
        let mut frame = vec![0u8; 5];
        frame[..4].copy_from_slice(&0u32.to_be_bytes());
        assert_eq!(
            split_frame(&frame).expect_err("zero"),
            FrameError::Truncated { have: 0, need: 1 }
        );
    }

    #[test]
    fn oversize_frames_rejected_on_both_sides() {
        // Encode side: the cap counts payload + flag byte.
        let big = vec![0u8; MAX_STREAM_FRAME_BYTES as usize + 1];
        let expected_len = big.len() as u64 + 1;
        let err = frame_payload(Ok(big)).expect_err("encode cap");
        assert_eq!(err, FrameError::TooLarge(expected_len));
        // Decode side: a forged oversize header is rejected before allocation.
        let mut forged = Vec::new();
        forged.extend_from_slice(&(MAX_STREAM_FRAME_BYTES + 1).to_be_bytes());
        forged.push(CODEC_FLAGS_V1);
        assert_eq!(
            split_frame(&forged).expect_err("decode cap"),
            FrameError::TooLarge(u64::from(MAX_STREAM_FRAME_BYTES) + 1)
        );
    }

    #[test]
    fn encode_failure_surfaces() {
        struct FailSer;
        impl serde::Serialize for FailSer {
            fn serialize<S: serde::Serializer>(&self, _s: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("deliberate"))
            }
        }
        assert_eq!(
            encode_frame(&FailSer).expect_err("encode"),
            FrameError::Encode
        );
    }

    #[test]
    fn decode_type_clash_errors() {
        // postcard is NOT self-describing: many type confusions decode "successfully"
        // (which is why the typed channel enums + TLV blobs exist above this layer).
        // A genuinely undecodable case: a String wanting more bytes than the payload has.
        let frame = encode_frame(&1u8).expect("encode");
        let err = decode_frame::<String>(&frame)
            .map(|_| ())
            .expect_err("clash");
        assert_eq!(err, FrameError::Decode);
    }

    proptest! {
        /// split_frame is total over arbitrary bytes.
        #[test]
        fn arbitrary_bytes_never_panic(bytes in proptest::collection::vec(any::<u8>(), 0..128)) {
            let _ = split_frame(&bytes);
        }

        /// Every encodable byte payload roundtrips.
        #[test]
        fn payload_roundtrips(payload in proptest::collection::vec(any::<u8>(), 0..512)) {
            let frame = encode_frame(&payload).expect("encode");
            let (back, consumed): (Vec<u8>, usize) = decode_frame(&frame).expect("decode");
            prop_assert_eq!(back, payload);
            prop_assert_eq!(consumed, frame.len());
        }
    }
}
