// HPACK (RFC 7541) header compression.
//
// Decoder: complete — indexed, literal (all three indexing modes), and dynamic
// table size updates, with a full dynamic table. A server may use any of these.
//
// Encoder: deliberately simple and stateless. It indexes exact static-table
// matches, references static names, and Huffman-encodes strings when that is
// shorter, but never inserts into a dynamic table. This is fully spec-compliant
// and keeps the encoder side free of mutable per-connection state.
import { ByteReader, ByteWriter, decodeUtf8, encodeUtf8 } from "../bytes.js";
import { H2Error } from "../errors.js";
import { huffmanDecode, huffmanEncode, huffmanShorter } from "./huffman.js";
import {
  STATIC_TABLE,
  STATIC_TABLE_LENGTH,
  type HeaderEntry,
} from "./static-table.js";

export interface Header {
  name: string;
  value: string;
  /** If set, encode as "never indexed" (0x10) — for sensitive values. */
  neverIndex?: boolean;
}

const ENTRY_OVERHEAD = 32; // RFC 7541 §4.1
const DEFAULT_TABLE_SIZE = 4096;

// --- Prefix integer coding (RFC 7541 §5.1) ---

/** Write `value` as a prefix integer; `firstByteFlags` occupies the high bits. */
function writeInteger(
  w: ByteWriter,
  value: number,
  prefixBits: number,
  firstByteFlags: number,
): void {
  const max = (1 << prefixBits) - 1;
  if (value < max) {
    w.u8(firstByteFlags | value);
    return;
  }
  w.u8(firstByteFlags | max);
  let rest = value - max;
  while (rest >= 128) {
    w.u8((rest % 128) + 128);
    rest = Math.floor(rest / 128);
  }
  w.u8(rest);
}

function readInteger(r: ByteReader, prefixBits: number): number {
  const max = (1 << prefixBits) - 1;
  let value = r.u8() & max;
  if (value < max) return value;
  let shift = 0;
  let byte: number;
  do {
    byte = r.u8();
    value += (byte & 0x7f) * 2 ** shift;
    shift += 7;
    if (shift > 42) throw new H2Error("COMPRESSION_ERROR", "integer overflow");
  } while (byte & 0x80);
  return value;
}

// --- String coding (RFC 7541 §5.2) ---

function writeString(w: ByteWriter, s: string): void {
  const raw = encodeUtf8(s);
  if (huffmanShorter(raw)) {
    const encoded = huffmanEncode(raw);
    writeInteger(w, encoded.length, 7, 0x80);
    w.bytes(encoded);
  } else {
    writeInteger(w, raw.length, 7, 0x00);
    w.bytes(raw);
  }
}

function readString(r: ByteReader): string {
  const huffman = (r.peek() & 0x80) !== 0;
  const length = readInteger(r, 7);
  if (length > r.remaining) {
    throw new H2Error("COMPRESSION_ERROR", "string length exceeds block");
  }
  const bytes = r.bytes(length);
  return decodeUtf8(huffman ? huffmanDecode(bytes) : bytes);
}

// --- Static table lookup maps (built once) ---

const staticNameToIndex = new Map<string, number>(); // 1-based, first occurrence
const staticPairToIndex = new Map<string, number>(); // 1-based
for (let i = 0; i < STATIC_TABLE.length; i++) {
  const [name, value] = STATIC_TABLE[i]!;
  const index = i + 1;
  if (!staticNameToIndex.has(name)) staticNameToIndex.set(name, index);
  staticPairToIndex.set(`${name}\0${value}`, index);
}

/**
 * HPACK decoder with a dynamic table. One instance per connection (inbound
 * compression context).
 */
export class HpackDecoder {
  private dynamic: HeaderEntry[] = []; // newest first
  private size = 0;
  private maxSize: number;
  private protocolMax: number;

  constructor(maxSize = DEFAULT_TABLE_SIZE) {
    this.maxSize = maxSize;
    this.protocolMax = maxSize;
  }

  /** The largest table size we've told the peer we can handle (our SETTINGS). */
  setProtocolMaxSize(n: number): void {
    this.protocolMax = n;
    if (this.maxSize > n) this.applyMaxSize(n);
  }

  private entryAt(index: number): HeaderEntry {
    if (index >= 1 && index <= STATIC_TABLE_LENGTH) {
      return STATIC_TABLE[index - 1]!;
    }
    const di = index - STATIC_TABLE_LENGTH - 1;
    const entry = this.dynamic[di];
    if (entry === undefined) {
      throw new H2Error("COMPRESSION_ERROR", `invalid HPACK index ${index}`);
    }
    return entry;
  }

  private insert(name: string, value: string): void {
    const entrySize = name.length + value.length + ENTRY_OVERHEAD;
    while (this.size + entrySize > this.maxSize && this.dynamic.length > 0) {
      const removed = this.dynamic.pop()!;
      this.size -= removed[0].length + removed[1].length + ENTRY_OVERHEAD;
    }
    if (entrySize <= this.maxSize) {
      this.dynamic.unshift([name, value]);
      this.size += entrySize;
    } else {
      // Entry larger than the whole table: the table ends up empty (§4.4).
      this.dynamic = [];
      this.size = 0;
    }
  }

  private applyMaxSize(newSize: number): void {
    if (newSize > this.protocolMax) {
      throw new H2Error("COMPRESSION_ERROR", "dynamic table size update too large");
    }
    this.maxSize = newSize;
    while (this.size > this.maxSize && this.dynamic.length > 0) {
      const removed = this.dynamic.pop()!;
      this.size -= removed[0].length + removed[1].length + ENTRY_OVERHEAD;
    }
  }

  /** Decode a complete header block fragment into a list of headers. */
  decode(block: Uint8Array): Header[] {
    const r = new ByteReader(block);
    const headers: Header[] = [];

    while (r.remaining > 0) {
      const first = r.peek();

      if (first & 0x80) {
        // Indexed header field.
        const index = readInteger(r, 7);
        if (index === 0) {
          throw new H2Error("COMPRESSION_ERROR", "indexed field with index 0");
        }
        const [name, value] = this.entryAt(index);
        headers.push({ name, value });
      } else if (first & 0x40) {
        // Literal with incremental indexing.
        const nameIndex = readInteger(r, 6);
        const name = nameIndex === 0 ? readString(r) : this.entryAt(nameIndex)[0];
        const value = readString(r);
        this.insert(name, value);
        headers.push({ name, value });
      } else if (first & 0x20) {
        // Dynamic table size update.
        this.applyMaxSize(readInteger(r, 5));
      } else {
        // Literal without indexing (0x00) or never indexed (0x10); prefix 4.
        const nameIndex = readInteger(r, 4);
        const name = nameIndex === 0 ? readString(r) : this.entryAt(nameIndex)[0];
        const value = readString(r);
        headers.push({ name, value });
      }
    }
    return headers;
  }
}

/**
 * HPACK encoder (outbound compression context). It never adds to the dynamic
 * table, so its only state is the table size it still owes the peer a signal for.
 */
export class HpackEncoder {
  private maxTableSize = DEFAULT_TABLE_SIZE;
  private pendingTableSizeUpdate: number | undefined;

  /** Apply the peer's SETTINGS_HEADER_TABLE_SIZE. A reduction must be signalled
   * at the start of the next header block (§4.2, §6.3). */
  setMaxTableSize(size: number): void {
    if (size >= this.maxTableSize) return;
    this.maxTableSize = size;
    this.pendingTableSizeUpdate = size;
  }

  encode(headers: readonly Header[]): Uint8Array {
    const w = new ByteWriter(256);
    if (this.pendingTableSizeUpdate !== undefined) {
      writeInteger(w, this.pendingTableSizeUpdate, 5, 0x20);
      this.pendingTableSizeUpdate = undefined;
    }
    for (const header of headers) {
      const name = header.name.toLowerCase();
      const value = header.value;

      const pair = staticPairToIndex.get(`${name}\0${value}`);
      if (pair !== undefined) {
        // Fully indexed static entry.
        writeInteger(w, pair, 7, 0x80);
        continue;
      }

      // Literal, no dynamic indexing. Never-indexed (0x10) for sensitive
      // headers, otherwise without-indexing (0x00). Both use a 4-bit prefix.
      const flags = header.neverIndex ? 0x10 : 0x00;
      const nameIndex = staticNameToIndex.get(name);
      if (nameIndex !== undefined) {
        writeInteger(w, nameIndex, 4, flags);
      } else {
        writeInteger(w, 0, 4, flags);
        writeString(w, name);
      }
      writeString(w, value);
    }
    return w.take().slice();
  }
}
