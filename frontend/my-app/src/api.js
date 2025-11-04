const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000";

const post = async (path, signal) => {
  try {
    const res = await fetch(`${API_BASE}/control/${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal,
      credentials: "include", // Include cookies for CORS
    });

    // Parse JSON response (even for error statuses)
    const contentType = res.headers.get("content-type") || "";
    const isJson = contentType.includes("application/json");
    
    let data = {};
    if (isJson) {
      try {
        data = await res.json();
      } catch (e) {
        // If JSON parsing fails, use empty object
        data = {};
      }
    }

    // If response is not OK, return error but don't throw
    if (!res.ok) {
      const errorMessage = data.error || data.message || `Server responded with ${res.status}`;
      return {
        error: errorMessage,
        status: res.status,
        ok: false,
        ...data, // Include other response data if any
      };
    }

    // Success response
    return {
      ...data,
      status: res.status,
      ok: true,
    };
  } catch (err) {
    // Network errors or fetch failures
    console.error(`[API ERROR] ${path}:`, err);
    return {
      error: err.message || "Network error",
      ok: false,
    };
  }
};

const API = {
  eye: {
    start: () => post("eye/start"),
    stop: () => post("eye/stop"),
    calibrate: () => post("eye/calibrate"),
  },
  voice: {
    start: () => post("voice/start"),
    stop: () => post("voice/stop"),
  },
  system: {
    volume_start: () => post("volume/control/start"),
    volume_stop: () => post("volume/control/stop"),
    screenshot_start: () => post("screenshot/control/start"),
    screenshot_stop: () => post("screenshot/control/stop"),
    both_start: () => post("volume_screenshot/control/start"),
    both_stop: () => post("volume_screenshot/control/stop"),
  },
};

export default API;