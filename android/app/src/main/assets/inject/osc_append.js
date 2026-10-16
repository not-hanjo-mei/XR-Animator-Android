
;// XRAA (XR Animator for Android) - VMC over native UDP instead of WebSocket
(function () {
  if (!self.XRANative || !self.OSC) return;
  class NativeUdpPlugin {
    constructor(options) {
      this.options = Object.assign({ host: '127.0.0.1', port: 39539 }, options || {});
      this.notify = () => {};
      this.socketStatus = -1;
      this.socket = { readyState: 3 };
    }
    registerNotify(fn) { this.notify = fn; }
    status() { return this.socketStatus; }
    open() {
      this.socketStatus = 1; this.socket.readyState = 1;
      XRANative.vmcOpen(String(this.options.host || '127.0.0.1'), (this.options.port | 0) || 39539);
      setTimeout(() => this.notify('open'), 0);
    }
    close() {
      this.socketStatus = 3; this.socket.readyState = 3;
      this.notify('close');
    }
    send(binary) {
      if (this.socketStatus != 1) return;
      let s = '';
      for (let i = 0; i < binary.length; i += 0x8000) s += String.fromCharCode.apply(null, binary.subarray(i, i + 0x8000));
      XRANative.vmcSend(btoa(s), String(this.options.host || '127.0.0.1'), (this.options.port | 0) || 39539);
    }
  }
  self.OSC.WebsocketClientPlugin = NativeUdpPlugin;
  console.log('(XRA native: VMC WebSocket sender replaced by native UDP)');
})();
