import { useState, useRef, useEffect } from "react";
import {
  View, Text, StyleSheet, TouchableOpacity, Image,
  ScrollView, ActivityIndicator, Dimensions,
  StatusBar, SafeAreaView, Alert, Switch,
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import * as Speech from "expo-speech";
import * as Location from "expo-location";
import { createClient } from "@supabase/supabase-js";

// ─── CONFIG ──────────────────────────────────────────────────────────────────
const HF_API_URL   = "http://localhost:7860";
const SUPABASE_URL = "https://fudxpmbtyhvmzchtuigu.supabase.co";
const SUPABASE_KEY = "sb_publishable_KCb9VD47biJxWBIglzoENA_t2Vohnb-";
const supabase     = createClient(SUPABASE_URL, SUPABASE_KEY);
const { width: SW } = Dimensions.get("window");

const ALERT_FAR      = 1300;
const ALERT_MED      = 500;
const ALERT_CLOSE    = 200;
const ALERT_DANGER   = 50;
const GPS_INTERVAL   = 3000;
const VOICE_COOLDOWN = 10000;

// ─── THEME ───────────────────────────────────────────────────────────────────
const C = {
  bg:        "#0a0a0f",
  card:      "#13131f",
  border:    "#1e1e30",
  accent:    "#3b82f6",
  success:   "#10b981",
  warning:   "#f59e0b",
  danger:    "#ef4444",
  textPri:   "#f1f5f9",
  textSec:   "#64748b",
  textMuted: "#334155",
};

// ─── HELPERS ─────────────────────────────────────────────────────────────────
const haversineM = (lat1, lon1, lat2, lon2) => {
  const R = 6371000;
  const toRad = x => x * Math.PI / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a = Math.sin(dLat/2)**2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
};

const formatDist = (m) => {
  if (m >= 1000) return `${(m / 1609).toFixed(1)} mi`;
  return `${Math.round(m)} m`;
};

const formatTime = (m, speedKmh = 50) => {
  const seconds = m / (speedKmh * 1000 / 3600);
  if (seconds < 60) return `${Math.round(seconds)} sec`;
  return `${Math.round(seconds / 60)} min`;
};

const severityColor = (s) => {
  if (s === "high")   return C.danger;
  if (s === "medium") return C.warning;
  return C.success;
};

const severityLabel = (conf) => {
  if (conf >= 0.85) return "high";
  if (conf >= 0.65) return "medium";
  return "low";
};

// ─── VOICE ───────────────────────────────────────────────────────────────────
const speak = (text) => {
  Speech.stop();
  Speech.speak(text, { language: "en-US", pitch: 1.0, rate: 0.88 });
};

// ─── SUPABASE ─────────────────────────────────────────────────────────────────
const fetchNearbyPotholes = async (lat, lon, radiusM = 5000) => {
  const { data, error } = await supabase
    .from("potholes")
    .select("*")
    .order("created_at", { ascending: false });
  if (error) { console.error("Supabase fetch:", error); return []; }
  return (data || [])
    .map(p => ({ ...p, dist: haversineM(lat, lon, p.lat, p.lon) }))
    .filter(p => p.dist <= radiusM)
    .sort((a, b) => a.dist - b.dist);
};

const reportPothole = async (lat, lon, severity, confidence, reportedBy = "community") => {
  const { error } = await supabase.from("potholes").insert([{
    lat, lon, severity, confidence, reported_by: reportedBy,
  }]);
  if (error) throw new Error(error.message);
};

// ─── GRADIO 6 API ─────────────────────────────────────────────────────────────
async function detectAndReport(imageUri, lat, lon, onStatus) {

  // Step 1: Upload
  onStatus("Uploading image...");
  const formData = new FormData();
  formData.append("files", { uri: imageUri, type: "image/jpeg", name: "road.jpg" });

  const uploadRes = await fetch(`${HF_API_URL}/gradio_api/upload`, {
    method: "POST",
    body  : formData,
  });
  if (!uploadRes.ok) throw new Error(`Upload failed: ${uploadRes.status}`);
  const uploadData   = await uploadRes.json();
  const uploadedPath = Array.isArray(uploadData) ? uploadData[0] : uploadData?.files?.[0];
  if (!uploadedPath) throw new Error("No path from upload");

  // Step 2: Call — Gradio 6 call/event_id pattern
  onStatus("Calling AI...");
  const callRes = await fetch(`${HF_API_URL}/gradio_api/call/detect_potholes`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({
      data: [
        { path: uploadedPath, meta: { _type: "gradio.FileData" } },
        0.25,
        lat && lon ? `${lat},${lon}` : "",
      [],
      ],
    }),
  });

  if (!callRes.ok) {
    const errText = await callRes.text();
    throw new Error(`Call failed ${callRes.status}: ${errText.slice(0, 150)}`);
  }

  // Parse event_id defensively
  const callText = await callRes.text();
  let event_id;
  try {
    const callJson = JSON.parse(callText);
    event_id = callJson.event_id;
  } catch (e) {
    throw new Error(`Could not parse call response: ${callText.slice(0, 150)}`);
  }
  if (!event_id) {
    throw new Error(`No event_id returned. Server said: ${callText.slice(0, 150)}`);
  }

  // Step 3: Poll for result
  onStatus("Analyzing...");
  // Small delay to let server process
  await new Promise(r => setTimeout(r, 500));

  const resultRes = await fetch(
    `${HF_API_URL}/gradio_api/call/detect_potholes/${event_id}`
  );
  if (!resultRes.ok) throw new Error(`Poll failed: ${resultRes.status}`);

  const rawText  = await resultRes.text();

  // Check for error event
  if (rawText.includes("event: error")) {
    throw new Error(`Server processing error. Check Gradio server logs.`);
  }

  const dataLine = rawText.split("\n").find(l => l.startsWith("data:"));
  if (!dataLine) throw new Error(`No data in response: ${rawText.slice(0, 150)}`);

  let parsed;
  try {
    parsed = JSON.parse(dataLine.replace("data: ", "").trim());
  } catch (e) {
    throw new Error(`Could not parse result: ${dataLine.slice(0, 150)}`);
  }

  const resultImage = parsed[0];
  const resultText  = parsed[1] ?? "";

  // Parse detections from markdown table
  const detections = [];
  const rows = resultText.split("\n").filter(r => r.startsWith("|") && /\d/.test(r));
  for (const row of rows) {
    const cells = row.split("|").map(c => c.trim()).filter(Boolean);
    if (cells.length >= 3) {
      const conf = parseFloat(cells[1]);
      if (!isNaN(conf)) {
        detections.push({
          confidence: conf,
          severity  : severityLabel(conf),
          bbox      : cells[2],
        });
      }
    }
  }

  // Log to Supabase
  if (detections.length > 0 && lat && lon) {
    onStatus("Logging to database...");
    const maxConf = Math.max(...detections.map(d => d.confidence));
    await reportPothole(lat, lon, severityLabel(maxConf), maxConf, "ai_detection");
  }

  return { resultImage, detections };
}

// ─── COMPONENTS ──────────────────────────────────────────────────────────────

function TabBar({ tab, setTab }) {
  const tabs = [
    { id: "navigator", label: "Navigate", icon: "🧭" },
    { id: "detect",    label: "Detect",   icon: "📷" },
    { id: "community", label: "Report",   icon: "📍" },
  ];
  return (
    <View style={s.tabBar}>
      {tabs.map(t => (
        <TouchableOpacity
          key={t.id}
          style={[s.tabBtn, tab === t.id && s.tabBtnActive]}
          onPress={() => setTab(t.id)}
          activeOpacity={0.75}
        >
          <Text style={s.tabIcon}>{t.icon}</Text>
          <Text style={[s.tabLabel, tab === t.id && s.tabLabelActive]}>{t.label}</Text>
        </TouchableOpacity>
      ))}
    </View>
  );
}

function AlertBanner({ nearest }) {
  if (!nearest) {
    return (
      <View style={[s.alertBanner, s.alertClear]}>
        <Text style={s.alertEmoji}>✅</Text>
        <View>
          <Text style={s.alertTitle}>All clear</Text>
          <Text style={s.alertSub}>No potholes detected nearby</Text>
        </View>
      </View>
    );
  }
  const d     = nearest.dist;
  const color = d < ALERT_CLOSE ? C.danger : d < ALERT_MED ? C.warning : C.accent;
  const emoji = d < ALERT_DANGER ? "🚨" : d < ALERT_CLOSE ? "⚠️" : "📍";
  return (
    <View style={[s.alertBanner, { backgroundColor: color + "18", borderColor: color + "55" }]}>
      <Text style={s.alertEmoji}>{emoji}</Text>
      <View style={{ flex: 1 }}>
        <Text style={[s.alertTitle, { color }]}>Pothole in {formatDist(d)}</Text>
        <Text style={s.alertSub}>~{formatTime(d)} ahead · {nearest.severity} severity</Text>
      </View>
      <View style={[s.distBadge, { backgroundColor: color + "22", borderColor: color + "44" }]}>
        <Text style={[s.distBadgeText, { color }]}>{formatDist(d)}</Text>
      </View>
    </View>
  );
}

// ─── NAVIGATOR TAB ───────────────────────────────────────────────────────────
function NavigatorTab() {
  const [location, setLocation]         = useState(null);
  const [gpsStatus, setGpsStatus]       = useState("Tap Start to begin monitoring");
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const [monitoring, setMonitoring]     = useState(false);
  const [nearest, setNearest]           = useState(null);
  const [allNearby, setAllNearby]       = useState([]);
  const [lastAlerts, setLastAlerts]     = useState({});
  const watchRef    = useRef(null);
  const intervalRef = useRef(null);

  const startMonitoring = async () => {
    const { status } = await Location.requestForegroundPermissionsAsync();
    if (status !== "granted") {
      Alert.alert("Permission needed", "Location access is required for navigation alerts.");
      return;
    }
    setMonitoring(true);
    setGpsStatus("GPS active — monitoring...");
    if (voiceEnabled) speak("Navigation mode activated. I will alert you when approaching a pothole.");
    watchRef.current = await Location.watchPositionAsync(
      { accuracy: Location.Accuracy.High, timeInterval: GPS_INTERVAL, distanceInterval: 5 },
      (loc) => setLocation(loc.coords)
    );
  };

  const stopMonitoring = () => {
    setMonitoring(false);
    setGpsStatus("Monitoring stopped");
    watchRef.current?.remove();
    clearInterval(intervalRef.current);
    if (voiceEnabled) speak("Navigation mode deactivated.");
  };

  useEffect(() => {
    if (!monitoring || !location) return;
    const check = async () => {
      const nearby = await fetchNearbyPotholes(location.latitude, location.longitude);
      setAllNearby(nearby);
      if (nearby.length === 0) { setNearest(null); return; }
      const closest = nearby[0];
      setNearest(closest);
      if (!voiceEnabled) return;
      const now  = Date.now();
      const last = lastAlerts[closest.id] || 0;
      if (now - last < VOICE_COOLDOWN) return;
      const d = closest.dist;
      let msg = null;
      if (d < ALERT_DANGER)     msg = "Danger! Pothole immediately ahead. Slow down now!";
      else if (d < ALERT_CLOSE) msg = `Warning! Pothole in ${formatDist(d)}. Prepare to slow down.`;
      else if (d < ALERT_MED)   msg = `Caution. Pothole ahead in ${formatDist(d)}, approximately ${formatTime(d)}.`;
      else if (d < ALERT_FAR)   msg = `Pothole detected in ${formatDist(d)}, about ${formatTime(d)} ahead.`;
      if (msg) {
        speak(msg);
        setLastAlerts(prev => ({ ...prev, [closest.id]: now }));
      }
    };
    intervalRef.current = setInterval(check, GPS_INTERVAL);
    return () => clearInterval(intervalRef.current);
  }, [monitoring, location, voiceEnabled]);

  return (
    <ScrollView contentContainerStyle={s.tabContent} showsVerticalScrollIndicator={false}>
      <View style={s.gpsBar}>
        <View style={[s.gpsDot, { backgroundColor: monitoring ? C.success : C.textMuted }]} />
        <Text style={s.gpsText}>{gpsStatus}</Text>
        {location && (
          <Text style={s.gpsCoords}>{location.latitude.toFixed(4)}, {location.longitude.toFixed(4)}</Text>
        )}
      </View>
      <AlertBanner nearest={nearest} />
      <View style={s.voiceCard}>
        <View style={s.voiceRow}>
          <View style={s.voiceLeft}>
            <Text style={s.voiceIcon}>🔊</Text>
            <View>
              <Text style={s.voiceTitle}>Voice alerts</Text>
              <Text style={s.voiceSub}>Speak distance warnings while driving</Text>
            </View>
          </View>
          <Switch
            value={voiceEnabled}
            onValueChange={setVoiceEnabled}
            trackColor={{ false: C.border, true: C.accent + "88" }}
            thumbColor={voiceEnabled ? C.accent : C.textSec}
          />
        </View>
      </View>
      <TouchableOpacity
        style={[s.bigBtn, monitoring ? s.bigBtnStop : s.bigBtnStart]}
        onPress={monitoring ? stopMonitoring : startMonitoring}
        activeOpacity={0.85}
      >
        <Text style={s.bigBtnIcon}>{monitoring ? "⏹" : "▶"}</Text>
        <Text style={s.bigBtnText}>{monitoring ? "Stop monitoring" : "Start monitoring"}</Text>
      </TouchableOpacity>
      {monitoring && (
        <View style={s.infoBox}>
          <Text style={s.infoText}>
            Keep this app open alongside Google Maps. Voice alerts will play automatically as you approach potholes.
          </Text>
        </View>
      )}
      {allNearby.length > 0 && (
        <View style={s.nearbyCard}>
          <Text style={s.nearbyTitle}>Nearby potholes ({allNearby.length})</Text>
          {allNearby.slice(0, 5).map((p, i) => (
            <View key={p.id} style={[s.nearbyRow, i === 0 && { borderTopWidth: 0 }]}>
              <View style={[s.nearbyDot, { backgroundColor: severityColor(p.severity) }]} />
              <View style={{ flex: 1 }}>
                <Text style={s.nearbyDist}>{formatDist(p.dist)} away</Text>
                <Text style={s.nearbyMeta}>~{formatTime(p.dist)} · {p.severity} · {p.reported_by}</Text>
              </View>
              <Text style={s.nearbyTime}>{new Date(p.created_at).toLocaleDateString()}</Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  );
}

// ─── DETECT TAB ──────────────────────────────────────────────────────────────
function DetectTab() {
  const [image, setImage]           = useState(null);
  const [loading, setLoading]       = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [result, setResult]         = useState(null);
  const [detections, setDetections] = useState([]);
  const [error, setError]           = useState("");
  const [location, setLocation]     = useState(null);

  useEffect(() => {
    Location.requestForegroundPermissionsAsync().then(({ status }) => {
      if (status === "granted") {
        Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High })
          .then(loc => setLocation(loc.coords))
          .catch(() => {});
      }
    });
  }, []);

  const pickImage = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (status !== "granted") { Alert.alert("Permission needed", "Allow photo library access."); return; }
    const res = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ["images"], quality: 0.85 });
    if (!res.canceled && res.assets?.[0]) { setImage(res.assets[0].uri); setError(""); setResult(null); }
  };

  const takePhoto = async () => {
    const { status } = await ImagePicker.requestCameraPermissionsAsync();
    if (status !== "granted") { Alert.alert("Permission needed", "Allow camera access."); return; }
    const res = await ImagePicker.launchCameraAsync({ quality: 0.85 });
    if (!res.canceled && res.assets?.[0]) { setImage(res.assets[0].uri); setError(""); setResult(null); }
  };

  const runDetection = async () => {
    if (!image) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const { resultImage, detections: dets } = await detectAndReport(
        image, location?.latitude, location?.longitude, setLoadingMsg
      );
      setResult(resultImage);
      setDetections(dets);
      if (dets.length > 0) {
        const high = dets.filter(d => d.severity === "high").length;
        speak(high > 0
          ? `${dets.length} potholes detected. ${high} high severity. Logged to community database.`
          : `${dets.length} potholes detected and logged.`
        );
      } else {
        speak("No potholes detected in this image.");
      }
    } catch (e) {
      console.error("Detection error:", e);
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const hasHigh = detections.some(d => d.severity === "high");

  return (
    <ScrollView contentContainerStyle={s.tabContent} showsVerticalScrollIndicator={false}>
      <View style={s.previewCard}>
        {image ? (
          <Image source={{ uri: image }} style={s.previewImage} resizeMode="cover" />
        ) : (
          <View style={s.previewEmpty}>
            <Text style={s.previewIcon2}>📷</Text>
            <Text style={s.previewEmptyTitle}>Take or upload a road photo</Text>
            <Text style={s.previewEmptyText}>AI detects potholes and logs to community map</Text>
          </View>
        )}
      </View>
      <View style={s.actionRow}>
        <TouchableOpacity style={s.actionBtn} onPress={takePhoto} activeOpacity={0.75}>
          <Text style={s.actionBtnIcon}>📷</Text>
          <Text style={s.actionBtnText}>Camera</Text>
        </TouchableOpacity>
        <TouchableOpacity style={s.actionBtn} onPress={pickImage} activeOpacity={0.75}>
          <Text style={s.actionBtnIcon}>🖼️</Text>
          <Text style={s.actionBtnText}>Gallery</Text>
        </TouchableOpacity>
      </View>
      {location && (
        <View style={s.gpsBar}>
          <View style={[s.gpsDot, { backgroundColor: C.success }]} />
          <Text style={s.gpsText}>GPS ready — detection will be logged to map</Text>
        </View>
      )}
      {error ? (
        <View style={s.errorBox}>
          <Text style={s.errorText}>{error}</Text>
        </View>
      ) : null}
      <TouchableOpacity
        style={[s.detectBtn, (!image || loading) && s.detectBtnDisabled]}
        onPress={runDetection}
        disabled={!image || loading}
        activeOpacity={0.85}
      >
        {loading ? (
          <View style={s.detectBtnInner}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={s.detectBtnText}>  {loadingMsg}</Text>
          </View>
        ) : (
          <Text style={s.detectBtnText}>{image ? "Detect & Log Potholes" : "Select an Image First"}</Text>
        )}
      </TouchableOpacity>
      {result && (
        <View style={s.resultCard}>
          <Image
            source={{ uri: result?.url ?? result }}
            style={s.resultImage}
            resizeMode="contain"
          />
          <View style={[s.summaryBanner,
            detections.length === 0 ? s.summaryGreen :
            hasHigh ? s.summaryRed : s.summaryAmber
          ]}>
            <Text style={s.summaryEmoji}>
              {detections.length === 0 ? "✅" : hasHigh ? "🚨" : "⚠️"}
            </Text>
            <Text style={s.summaryCount}>
              {detections.length === 0 ? "No potholes detected" : `${detections.length} potholes found`}
            </Text>
          </View>
          {location && detections.length > 0 && (
            <Text style={s.loggedText}>📍 Logged to community database</Text>
          )}
        </View>
      )}
    </ScrollView>
  );
}

// ─── COMMUNITY TAB ───────────────────────────────────────────────────────────
function CommunityTab() {
  const [reports, setReports]       = useState([]);
  const [loading, setLoading]       = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [location, setLocation]     = useState(null);
  const [severity, setSeverity]     = useState("medium");

  useEffect(() => {
    loadReports();
    Location.requestForegroundPermissionsAsync().then(({ status }) => {
      if (status === "granted") {
        Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.High })
          .then(loc => setLocation(loc.coords))
          .catch(() => {});
      }
    });
  }, []);

  const loadReports = async () => {
    setLoading(true);
    const { data } = await supabase
      .from("potholes")
      .select("*")
      .order("created_at", { ascending: false })
      .limit(20);
    setReports(data || []);
    setLoading(false);
  };

  const submitReport = async () => {
    if (!location) {
      Alert.alert("GPS needed", "Enable location to report a pothole at your current position.");
      return;
    }
    setSubmitting(true);
    try {
      await reportPothole(location.latitude, location.longitude, severity, 0.8, "community");
      speak("Pothole reported. Thank you for making the roads safer.");
      Alert.alert("Reported!", "Pothole logged to the community database. All drivers nearby will be alerted.");
      loadReports();
    } catch (e) {
      Alert.alert("Error", e.message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={s.tabContent} showsVerticalScrollIndicator={false}>
      <View style={s.reportCard}>
        <Text style={s.reportTitle}>Report a pothole here</Text>
        <Text style={s.reportSub}>Logs your current GPS location to the community database</Text>
        {location ? (
          <View style={[s.gpsBar, { marginTop: 12 }]}>
            <View style={[s.gpsDot, { backgroundColor: C.success }]} />
            <Text style={s.gpsText}>{location.latitude.toFixed(5)}, {location.longitude.toFixed(5)}</Text>
          </View>
        ) : (
          <View style={[s.gpsBar, { marginTop: 12 }]}>
            <View style={[s.gpsDot, { backgroundColor: C.warning }]} />
            <Text style={s.gpsText}>Waiting for GPS...</Text>
          </View>
        )}
        <Text style={[s.reportSub, { marginTop: 14, marginBottom: 8 }]}>Severity:</Text>
        <View style={s.severityRow}>
          {["low", "medium", "high"].map(sv => (
            <TouchableOpacity
              key={sv}
              style={[s.severityBtn, severity === sv && {
                backgroundColor: severityColor(sv) + "33",
                borderColor: severityColor(sv),
              }]}
              onPress={() => setSeverity(sv)}
            >
              <Text style={[s.severityText, severity === sv && { color: severityColor(sv) }]}>
                {sv.charAt(0).toUpperCase() + sv.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
        <TouchableOpacity
          style={[s.detectBtn, { marginTop: 16 }, submitting && s.detectBtnDisabled]}
          onPress={submitReport}
          disabled={submitting}
          activeOpacity={0.85}
        >
          {submitting ? (
            <View style={s.detectBtnInner}>
              <ActivityIndicator color="#fff" size="small" />
              <Text style={s.detectBtnText}>  Submitting...</Text>
            </View>
          ) : (
            <Text style={s.detectBtnText}>📍 Report Pothole Here</Text>
          )}
        </TouchableOpacity>
      </View>
      <View style={s.feedHeader}>
        <Text style={s.feedTitle}>Community reports</Text>
        <TouchableOpacity onPress={loadReports}>
          <Text style={s.refreshBtn}>Refresh</Text>
        </TouchableOpacity>
      </View>
      {loading ? (
        <ActivityIndicator color={C.accent} style={{ marginTop: 20 }} />
      ) : reports.length === 0 ? (
        <Text style={s.emptyText}>No reports yet — be the first!</Text>
      ) : (
        reports.map((r, i) => (
          <View key={r.id} style={s.feedItem}>
            <View style={[s.feedDot, { backgroundColor: severityColor(r.severity) }]} />
            <View style={{ flex: 1 }}>
              <Text style={s.feedLoc}>{r.lat.toFixed(4)}, {r.lon.toFixed(4)}</Text>
              <Text style={s.feedMeta}>
                {r.severity} · {r.reported_by} · {new Date(r.created_at).toLocaleDateString()}
              </Text>
            </View>
            <View style={[s.feedSeverity, {
              backgroundColor: severityColor(r.severity) + "22",
              borderColor: severityColor(r.severity) + "44",
            }]}>
              <Text style={[s.feedSeverityText, { color: severityColor(r.severity) }]}>
                {r.severity}
              </Text>
            </View>
          </View>
        ))
      )}
    </ScrollView>
  );
}

// ─── MAIN APP ────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState("navigator");
  return (
    <SafeAreaView style={s.root}>
      <StatusBar barStyle="light-content" backgroundColor={C.bg} />
      <View style={s.appHeader}>
        <View style={s.logoBox}>
          <Text style={s.logoText}>PA</Text>
        </View>
        <View>
          <Text style={s.appTitle}>Pothole Navigator</Text>
          <Text style={s.appSubtitle}>Community road safety · YOLOv8n AI</Text>
        </View>
      </View>
      <View style={{ flex: 1 }}>
        {tab === "navigator" && <NavigatorTab />}
        {tab === "detect"    && <DetectTab />}
        {tab === "community" && <CommunityTab />}
      </View>
      <TabBar tab={tab} setTab={setTab} />
    </SafeAreaView>
  );
}

// ─── STYLES ──────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root:              { flex: 1, backgroundColor: C.bg },
  appHeader:         { flexDirection: "row", alignItems: "center", gap: 12, paddingHorizontal: 20, paddingTop: 12, paddingBottom: 12 },
  logoBox:           { width: 38, height: 38, borderRadius: 11, backgroundColor: C.accent, alignItems: "center", justifyContent: "center" },
  logoText:          { color: "#fff", fontWeight: "800", fontSize: 13 },
  appTitle:          { color: C.textPri, fontSize: 16, fontWeight: "700" },
  appSubtitle:       { color: C.textSec, fontSize: 11, marginTop: 1 },
  tabBar:            { flexDirection: "row", backgroundColor: C.card, borderTopWidth: 1, borderTopColor: C.border, paddingBottom: 8, paddingTop: 6 },
  tabBtn:            { flex: 1, alignItems: "center", paddingVertical: 4, gap: 2 },
  tabBtnActive:      { borderTopWidth: 2, borderTopColor: C.accent, marginTop: -1 },
  tabIcon:           { fontSize: 20 },
  tabLabel:          { color: C.textSec, fontSize: 11 },
  tabLabelActive:    { color: C.accent, fontWeight: "600" },
  tabContent:        { paddingHorizontal: 16, paddingTop: 12, paddingBottom: 32 },
  gpsBar:            { flexDirection: "row", alignItems: "center", gap: 8, backgroundColor: C.card, borderWidth: 1, borderColor: C.border, borderRadius: 10, padding: 10, marginBottom: 12 },
  gpsDot:            { width: 8, height: 8, borderRadius: 4 },
  gpsText:           { color: C.textSec, fontSize: 12, flex: 1 },
  gpsCoords:         { color: C.textMuted, fontSize: 10 },
  alertBanner:       { flexDirection: "row", alignItems: "center", gap: 12, borderRadius: 14, padding: 14, marginBottom: 12, borderWidth: 1 },
  alertClear:        { backgroundColor: C.success + "12", borderColor: C.success + "33" },
  alertEmoji:        { fontSize: 26 },
  alertTitle:        { color: C.textPri, fontSize: 16, fontWeight: "700" },
  alertSub:          { color: C.textSec, fontSize: 12, marginTop: 2 },
  distBadge:         { borderRadius: 8, borderWidth: 1, paddingHorizontal: 10, paddingVertical: 6, alignItems: "center" },
  distBadgeText:     { fontSize: 14, fontWeight: "800" },
  voiceCard:         { backgroundColor: C.card, borderWidth: 1, borderColor: C.border, borderRadius: 12, marginBottom: 12 },
  voiceRow:          { flexDirection: "row", alignItems: "center", justifyContent: "space-between", padding: 12 },
  voiceLeft:         { flexDirection: "row", alignItems: "center", gap: 10 },
  voiceIcon:         { fontSize: 20 },
  voiceTitle:        { color: C.textPri, fontSize: 13, fontWeight: "600" },
  voiceSub:          { color: C.textSec, fontSize: 11, marginTop: 1 },
  bigBtn:            { borderRadius: 14, paddingVertical: 16, flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10, marginBottom: 12 },
  bigBtnStart:       { backgroundColor: C.success },
  bigBtnStop:        { backgroundColor: C.danger + "cc" },
  bigBtnIcon:        { fontSize: 18, color: "#fff" },
  bigBtnText:        { color: "#fff", fontSize: 16, fontWeight: "700" },
  infoBox:           { backgroundColor: C.accent + "12", borderWidth: 1, borderColor: C.accent + "33", borderRadius: 10, padding: 10, marginBottom: 12 },
  infoText:          { color: C.accent, fontSize: 12, lineHeight: 18 },
  nearbyCard:        { backgroundColor: C.card, borderWidth: 1, borderColor: C.border, borderRadius: 14, padding: 14, marginBottom: 12 },
  nearbyTitle:       { color: C.textPri, fontSize: 14, fontWeight: "700", marginBottom: 10 },
  nearbyRow:         { flexDirection: "row", alignItems: "center", gap: 10, paddingVertical: 8, borderTopWidth: 1, borderTopColor: C.border },
  nearbyDot:         { width: 8, height: 8, borderRadius: 4 },
  nearbyDist:        { color: C.textPri, fontSize: 13, fontWeight: "600" },
  nearbyMeta:        { color: C.textSec, fontSize: 11, marginTop: 1 },
  nearbyTime:        { color: C.textMuted, fontSize: 10 },
  previewCard:       { width: "100%", height: SW * 0.55, borderRadius: 16, overflow: "hidden", backgroundColor: C.card, borderWidth: 1, borderColor: C.border, marginBottom: 12 },
  previewImage:      { width: "100%", height: "100%" },
  previewEmpty:      { flex: 1, alignItems: "center", justifyContent: "center", gap: 8 },
  previewIcon2:      { fontSize: 36 },
  previewEmptyTitle: { color: C.textPri, fontSize: 14, fontWeight: "600" },
  previewEmptyText:  { color: C.textSec, fontSize: 12 },
  actionRow:         { flexDirection: "row", gap: 10, marginBottom: 12 },
  actionBtn:         { flex: 1, backgroundColor: C.card, borderWidth: 1, borderColor: C.border, borderRadius: 12, paddingVertical: 12, alignItems: "center", gap: 4 },
  actionBtnIcon:     { fontSize: 20 },
  actionBtnText:     { color: C.textPri, fontSize: 12, fontWeight: "600" },
  errorBox:          { backgroundColor: C.danger + "18", borderWidth: 1, borderColor: C.danger + "44", borderRadius: 10, padding: 10, marginBottom: 12 },
  errorText:         { color: C.danger, fontSize: 12, lineHeight: 18 },
  detectBtn:         { backgroundColor: C.accent, borderRadius: 14, paddingVertical: 15, alignItems: "center", marginBottom: 12 },
  detectBtnDisabled: { backgroundColor: C.textMuted },
  detectBtnInner:    { flexDirection: "row", alignItems: "center" },
  detectBtnText:     { color: "#fff", fontSize: 15, fontWeight: "700" },
  resultCard:        { borderRadius: 14, overflow: "hidden", backgroundColor: C.card, borderWidth: 1, borderColor: C.border, marginBottom: 12 },
  resultImage:       { width: "100%", height: SW * 0.6 },
  summaryBanner:     { flexDirection: "row", alignItems: "center", gap: 10, padding: 12, borderTopWidth: 1, borderTopColor: C.border },
  summaryGreen:      { backgroundColor: C.success + "18" },
  summaryAmber:      { backgroundColor: C.warning + "18" },
  summaryRed:        { backgroundColor: C.danger  + "18" },
  summaryEmoji:      { fontSize: 20 },
  summaryCount:      { color: C.textPri, fontSize: 14, fontWeight: "700" },
  loggedText:        { color: C.success, fontSize: 12, padding: 10, paddingTop: 0 },
  reportCard:        { backgroundColor: C.card, borderWidth: 1, borderColor: C.border, borderRadius: 14, padding: 14, marginBottom: 14 },
  reportTitle:       { color: C.textPri, fontSize: 15, fontWeight: "700" },
  reportSub:         { color: C.textSec, fontSize: 12, marginTop: 4 },
  severityRow:       { flexDirection: "row", gap: 8 },
  severityBtn:       { flex: 1, borderWidth: 1, borderColor: C.border, borderRadius: 8, paddingVertical: 8, alignItems: "center" },
  severityText:      { color: C.textSec, fontSize: 13, fontWeight: "600" },
  feedHeader:        { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 10 },
  feedTitle:         { color: C.textPri, fontSize: 14, fontWeight: "700" },
  refreshBtn:        { color: C.accent, fontSize: 13 },
  emptyText:         { color: C.textSec, fontSize: 13, textAlign: "center", marginTop: 20 },
  feedItem:          { flexDirection: "row", alignItems: "center", gap: 10, backgroundColor: C.card, borderWidth: 1, borderColor: C.border, borderRadius: 10, padding: 12, marginBottom: 8 },
  feedDot:           { width: 8, height: 8, borderRadius: 4 },
  feedLoc:           { color: C.textPri, fontSize: 12, fontWeight: "600" },
  feedMeta:          { color: C.textSec, fontSize: 11, marginTop: 2 },
  feedSeverity:      { borderRadius: 6, borderWidth: 1, paddingHorizontal: 8, paddingVertical: 3 },
  feedSeverityText:  { fontSize: 10, fontWeight: "700" },
});