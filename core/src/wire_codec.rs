//! Wire codec for length-prefixed messages with optional LZ4 compression.
//!
//! Wire format:
//!   [4 bytes: total_len (big-endian)] [1 byte: flags] [total_len-1 bytes: payload]
//!
//! Flags byte bit 0: payload is LZ4 compressed.
//! When compressed, payload = [4 bytes: uncompressed_len (BE)] [lz4 data].
//!
//! Only compresses payloads > COMPRESS_THRESHOLD bytes to avoid overhead
//! on small messages (e.g., PlayerInput).

/// Minimum payload size (before flags byte) to trigger compression.
const COMPRESS_THRESHOLD: usize = 128;

const FLAG_LZ4: u8 = 0x01;

/// Encode a serialized message into the wire format with optional LZ4 compression.
/// Appends the result to `out` (which should be cleared by the caller if needed).
pub fn encode(data: &[u8], out: &mut Vec<u8>) {
    if data.len() >= COMPRESS_THRESHOLD {
        let compressed = lz4_flex::compress_prepend_size(data);
        // Only use compression if it actually saves space.
        if compressed.len() < data.len() {
            let total_len = 1 + compressed.len(); // flags + compressed payload
            out.reserve(4 + total_len);
            out.extend_from_slice(&(total_len as u32).to_be_bytes());
            out.push(FLAG_LZ4);
            out.extend_from_slice(&compressed);
            return;
        }
    }
    // Uncompressed: flags = 0.
    let total_len = 1 + data.len();
    out.reserve(4 + total_len);
    out.extend_from_slice(&(total_len as u32).to_be_bytes());
    out.push(0u8);
    out.extend_from_slice(data);
}

/// Decode a wire-format payload (after reading the 4-byte length prefix).
/// `payload` should be the `total_len` bytes after the length prefix.
/// Returns the decompressed serialized message.
pub fn decode(payload: &[u8]) -> Result<Vec<u8>, DecodeError> {
    if payload.is_empty() {
        return Err(DecodeError::TooShort);
    }
    let flags = payload[0];
    let body = &payload[1..];

    if flags & FLAG_LZ4 != 0 {
        lz4_flex::decompress_size_prepended(body)
            .map_err(|e| DecodeError::Lz4(e.to_string()))
    } else {
        Ok(body.to_vec())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DecodeError {
    #[error("payload too short")]
    TooShort,
    #[error("LZ4 decompression failed: {0}")]
    Lz4(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_small_uncompressed() {
        let data = b"hello";
        let mut buf = Vec::new();
        encode(data, &mut buf);
        // Should be uncompressed: [4 len][1 flags=0][5 data]
        assert_eq!(buf[4], 0u8); // flags = no compression
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let decoded = decode(&buf[4..4 + len]).unwrap();
        assert_eq!(&decoded, data);
    }

    #[test]
    fn roundtrip_large_compressed() {
        // Repetitive data compresses well.
        let data = vec![42u8; 1024];
        let mut buf = Vec::new();
        encode(&data, &mut buf);
        // Should be compressed (highly repetitive).
        assert_eq!(buf[4] & FLAG_LZ4, FLAG_LZ4);
        // Total wire size should be smaller than uncompressed.
        assert!(buf.len() < 4 + 1 + data.len());
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let decoded = decode(&buf[4..4 + len]).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn incompressible_stays_raw() {
        // Random-ish data that won't compress.
        let data: Vec<u8> = (0..200).map(|i| (i * 37 + 13) as u8).collect();
        let mut buf = Vec::new();
        encode(&data, &mut buf);
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let decoded = decode(&buf[4..4 + len]).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn below_threshold_never_compressed() {
        // Data just under COMPRESS_THRESHOLD — must stay uncompressed even if compressible.
        let data = vec![0u8; COMPRESS_THRESHOLD - 1];
        let mut buf = Vec::new();
        encode(&data, &mut buf);
        assert_eq!(buf[4], 0u8); // flags = no compression
    }

    #[test]
    fn at_threshold_may_compress() {
        // Data exactly at COMPRESS_THRESHOLD and highly repetitive.
        let data = vec![0u8; COMPRESS_THRESHOLD];
        let mut buf = Vec::new();
        encode(&data, &mut buf);
        // Should compress (all zeros compress very well).
        assert_eq!(buf[4] & FLAG_LZ4, FLAG_LZ4);
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let decoded = decode(&buf[4..4 + len]).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn empty_data_roundtrips() {
        let data = b"";
        let mut buf = Vec::new();
        encode(data, &mut buf);
        let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
        let decoded = decode(&buf[4..4 + len]).unwrap();
        assert_eq!(&decoded, data);
    }

    #[test]
    fn decode_empty_payload_is_error() {
        assert!(decode(&[]).is_err());
    }

    #[test]
    fn buffer_reuse_works() {
        let mut buf = Vec::new();
        for i in 0..5 {
            let data = vec![i as u8; 200];
            buf.clear();
            encode(&data, &mut buf);
            let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
            let decoded = decode(&buf[4..4 + len]).unwrap();
            assert_eq!(decoded, data);
        }
    }
}
