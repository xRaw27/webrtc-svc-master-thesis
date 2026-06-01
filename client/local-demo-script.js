/**
 * webrtcperf automation script for the local LiveKit demo app
 * (client-sdk-js/examples/demo/).
 *
 * The URL handler pre-fills ?url=...&token=... in the form.
 * This script selects the codec, clicks Connect, then enables the camera.
 * Audio is intentionally left disabled — these scenarios test video SVC
 * layer allocation only.
 *
 * scriptParams:
 *   codec      – 'vp9' (default) or 'av1'
 *   enableCam  – session range, e.g. '0-5' (default: all)
 */

(async () => {
  const params = typeof webrtcperf !== 'undefined' ? webrtcperf.params : {};
  const codec = params.codec || 'vp9';

  // Wait for the page to load and the codec dropdown to be populated
  await new Promise(r => setTimeout(r, 1000));

  // --- 0. Disable AdaptiveStream (it reduces quality based on video element size) ---
  const adaptiveStreamCheckbox = document.getElementById('adaptive-stream');
  if (adaptiveStreamCheckbox && adaptiveStreamCheckbox.checked) {
    adaptiveStreamCheckbox.click();
    console.log('[webrtcperf] Disabled AdaptiveStream');
  }

  // --- 1. Select preferred codec ---
  const codecSelect = document.getElementById('preferred-codec');
  if (codecSelect) {
    // The options are populated dynamically by the demo app
    // Wait a bit more for them to appear
    let attempts = 0;
    while (codecSelect.options.length <= 1 && attempts < 10) {
      await new Promise(r => setTimeout(r, 500));
      attempts++;
    }
    // Find and select the desired codec
    for (let i = 0; i < codecSelect.options.length; i++) {
      if (codecSelect.options[i].value.toLowerCase() === codec.toLowerCase()) {
        codecSelect.selectedIndex = i;
        codecSelect.dispatchEvent(new Event('change', { bubbles: true }));
        console.log(`[webrtcperf] Selected codec: ${codec}`);
        break;
      }
    }
  }

  // Wait for scalability mode dropdown to enable (it enables when SVC codec is selected)
  await new Promise(r => setTimeout(r, 500));

  // --- 2. Set scalability mode to L3T3 ---
  const scalabilitySelect = document.getElementById('scalability-mode');
  if (scalabilitySelect) {
    for (let i = 0; i < scalabilitySelect.options.length; i++) {
      if (scalabilitySelect.options[i].value === 'L3T3') {
        scalabilitySelect.selectedIndex = i;
        scalabilitySelect.dispatchEvent(new Event('change', { bubbles: true }));
        console.log('[webrtcperf] Selected scalability mode: L3T3');
        break;
      }
    }
  }

  // --- 3. Click Connect ---
  const connectBtn = document.getElementById('connect-button');
  if (connectBtn) {
    connectBtn.click();
    console.log('[webrtcperf] Clicked Connect');
  }

  // Wait for connection to establish
  await new Promise(r => setTimeout(r, 3000));

  // --- 4. Disable Microphone (demo auto-enables it on connect when Publish is checked) ---
  const audioBtn = document.getElementById('toggle-audio-button');
  if (audioBtn && !audioBtn.disabled && audioBtn.textContent.trim().toLowerCase().startsWith('disable')) {
    audioBtn.click();
    console.log('[webrtcperf] Disabled microphone');
  }

  // --- 5. Enable Camera ---
  const shouldEnableCam = shouldEnableForSession(params.enableCam);
  if (shouldEnableCam) {
    const camBtn = document.getElementById('toggle-video-button');
    if (camBtn && !camBtn.disabled) {
      camBtn.click();
      console.log('[webrtcperf] Enabled camera');
    }
  }
})();

/**
 * Check if the current session index falls within the given range string.
 * Range format: "0-5", "0,2,4", "0-3,5" or undefined (= all sessions).
 */
function shouldEnableForSession(rangeStr) {
  if (rangeStr === undefined || rangeStr === null || rangeStr === '') return true;

  // Get current session index from webrtcperf
  const sessionIndex = typeof webrtcperf !== 'undefined' ? webrtcperf.index : 0;

  const parts = String(rangeStr).split(',');
  for (const part of parts) {
    if (part.includes('-')) {
      const [start, end] = part.split('-').map(Number);
      if (sessionIndex >= start && sessionIndex <= end) return true;
    } else {
      if (sessionIndex === Number(part)) return true;
    }
  }
  return false;
}
