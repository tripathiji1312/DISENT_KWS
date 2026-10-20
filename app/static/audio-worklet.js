/**
 * CaptureProcessor batches 128-frame render quanta into 2048-sample chunks.
 * The main thread sends each Float32Array as a binary WebSocket frame.
 */
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buffer = new Float32Array(2048);
    this.offset = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const channel = input[0];

    let read = 0;
    while (read < channel.length) {
      const writable = Math.min(channel.length - read, this.buffer.length - this.offset);
      this.buffer.set(channel.subarray(read, read + writable), this.offset);
      this.offset += writable;
      read += writable;

      if (this.offset === this.buffer.length) {
        const chunk = this.buffer.slice();
        this.port.postMessage(chunk, [chunk.buffer]);
        this.offset = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture-processor", CaptureProcessor);
