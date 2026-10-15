# 3D 職業車手圈速睇板（子項目 #1）— 設計

- 日期：2026-10-15
- 狀態：設計已批，待寫 implementation plan
- 靈感來源：Instagram「跟著 F1 高手學，你能駕快多少？EP2」— 3D 遊戲入面跟住職業車手 telemetry ghost 學 brake／racing line

## 1. 目標

喺現有 web app 加一個新 mode「3D 學習」：揀年份／車手／賽道之後，喺 3D 賽道上播放職業車手嘅排位圈，睇到佢喺每個彎幾時 brake、幾時收油、幾時再踩油，以及轉向、檔位、速度變化。

**唔使上載 CSV** —— 只用職業車手 reference lap。

### 成個計劃嘅分拆（只有 #1 喺今次範圍）

| # | 子項目 | 依賴 |
| --- | --- | --- |
| **1** | **3D 播放引擎 + 睇職業車手（本 spec）** | — |
| 2 | 3D 對比模式：你嘅 lap vs 職業車手，兩架車 + live delta | #1 |
| 3 | 節奏 brake 遊戲：車自己行，使用者喺 brake 點撳掣，判 Perfect／Good／Miss | #1 |
| 4 | 真正揸車：鍵盤控制 + 簡化物理 + racing line 指引 + ghost | #1 |

### 唔包（Non-goals）

- #2–#4 嘅任何功能
- 看台、樹木、路邊建築等賽道佈景；真實賽道闊度
- 任何 F1／車隊官方 logo、塗裝、車模型
- 改動 `src/`（現有規則：web 改動只喺 `app/`）

## 2. 數據事實（已驗證）

- `data/reference_laps/*.pkl` = `(DataFrame, meta)`；共 25 條 2025 Q 圈（NOR×20、VER×3、LEC×1、SAI×1）
- DataFrame 欄位：`lap_distance`（每 1 m 一行）、`speed_kmh`、`throttle`（0–1）、`brake`（0/1 boolean）、`gear`、`rpm`、`drs`、`world_position_X/Y/Z`
- **座標單位唔係米**：Bahrain 路徑長度 53,312 vs 賽道 5,372 m → FastF1 1/10 m。X/Y = 水平，Z = 高度
- `meta` 有 `lap_time`、`track_length`、`driver`、`year`、`gp_name`
- 冇 steering 數據（`app/main.py` 現有 `_ref_steering_from_curvature` 係由曲率推算）
- `tracks/*.json` 全部 24 條都有 `corners[].{id, short, name, entry_m, apex_m, exit_m, direction}`；只有 4 條有 `brake_start_m` → **brake 點唔用 json，由 lap 數據計**
- 冇 cache 嘅組合可經 `src.fastf1_loader.load_real_telemetry(driver, year, session, track)` 經 FastF1 下載並自動存 pkl

## 3. 架構

```text
UI：mode「3D 學習」→ 年份／車手／賽道（隱藏 CSV 上載同 lap 揀選）
        │  GET /pro_lap?year=2025&driver=NOR&track=bahrain&session=Q
        ▼
app/main.py  /pro_lap（新）
  ├ load_real_telemetry()                    ← 現有
  ├ 座標 → 米：scale = track_length / 路徑長度（逐條計，唔寫死 0.1）
  ├ XY 平滑（重用 _smooth_loop_xy）
  ├ 每 2 m 取一點
  ├ steer：由平滑後路徑曲率推算（同 _ref_steering_from_curvature 同一方法）
  ├ t：累計 Σ ds / v，再按比例縮放令最後一點 = meta.lap_time
  └ corners：json 嘅彎道 + 由 lap 計出嘅 brake 數據
        ▼
app/static/js/learn3d.js（新 ES module，Three.js 經 CDN importmap 載入）
  ├ scene    賽道路面 ribbon（中線兩邊各 7 m，跟 Z 高低）、草地平面、起點線、燈光
  ├ car      自己砌嘅 low-poly 開輪車（box + cylinder），中性顏色，冇 logo
  ├ playback 時間 t → 二分搜尋 points → 插值位置／方向（方向 = 路徑切線）
  ├ cameras  T-cam（車頂後上方第一身）／Chase（車後上方）／Top（俯視跟車）
  ├ markers  每個彎：地面紅色 BRAKE 線 + 黃色 apex 點
  └ panels   📈 遙測（下方曲線）／🏁 彎道（右邊清單），切換
```

### 改動檔案

| 檔案 | 改動 |
| --- | --- |
| `app/main.py` | 新 `GET /pro_lap`。`learn_3d` 係純前端 mode，**唔加入** `ALL_MODES`，`/analyze` 唔使改 |
| `app/static/index.html` | 新 mode 按鈕、3D 區塊 DOM、Three.js importmap |
| `app/static/js/learn3d.js` | 新：scene、車、播放、鏡頭、panels |
| `app/static/js/main.js` | 揀 `learn_3d` 時隱藏 CSV／lap 區、改為呼叫 `learn3d.load()` |
| `app/static/css/main.css` | 3D 區塊、HUD、panels、手機排版 |
| `tests/test_pro_lap.py` | 新 |

## 4. `/pro_lap` API

**Request**：`GET /pro_lap?year=<int>&driver=<code>&track=<short>&session=Q`

**Response 200**：

```json
{
  "meta": {"driver": "NOR", "year": 2025, "session": "Q", "gp_name": "...",
           "track": "bahrain", "track_name": "...", "lap_time": 90.267, "length_m": 5372},
  "points": {
    "d": [0, 2, 4, ...],          // lap distance m
    "x": [...], "y": [...], "z": [...],   // 米，已平滑
    "t": [...],                   // 秒，單調遞增
    "speed": [...], "throttle": [...], "brake": [...], "gear": [...],
    "steer": [...]                // 度，推算，左負右正
  },
  "corners": [
    {"id": 4, "short": "T4", "name": "...", "direction": "right",
     "entry_m": 1420, "apex_m": 1510, "exit_m": 1600,
     "action": "brake",           // "brake" | "lift" | "flat"
     "brake_start_m": 1380,       // action != brake 時為 null
     "brake_duration_s": 1.4,     // 同上
     "min_speed_kmh": 118, "apex_gear": 3}
  ]
}
```

- 用 column 陣列（唔係 object 陣列）減少體積；~2,700 點 × 10 欄 ≈ 150 KB

### 彎道數據計法

- 搜尋窗：`[entry_m − 250, apex_m]`
- `brake`：窗內有 `brake == 1` → `brake_start_m` = 第一個 brake 點；`brake_duration_s` = 由嗰點起連續 brake 嘅時間
- `lift`：冇 brake，但窗內 throttle 跌穿 0.9
- `flat`：以上都冇
- `min_speed_kmh`、`apex_gear`：`[entry_m, exit_m]` 內最低速嗰點

### Errors

| 情況 | Response |
| --- | --- |
| track 唔喺 `tracks/` | 404 `{"detail": "unknown track"}` |
| 冇 cache 且 FastF1 失敗 | 404 `{"detail": "no data", "available": [{"year","driver","track"}...]}`（列出現有 pkl） |
| 參數格式錯 | 422（FastAPI 預設） |

## 5. UI 同互動

### Layout（已用 mockup 確認）

- 上：3D 視窗，HUD 用 B 版簡潔款 —— 左上車手／圈速、「T4 · BRAKE in 82 m」倒數（下一個彎前 300 m 內先出），左下速度＋檔位＋油門／brake 柱，右上小地圖，右下鏡頭切換
- 頂部切換掣：**📈 遙測**（預設，B）／**🏁 彎道**（C）
  - 📈 遙測：3D 下方 Speed／Throttle／Brake／Steer（標「推算」）四條曲線，白色游標同步；點曲線 → 跳去該位置
  - 🏁 彎道：3D 右邊清單（彎、brake 點→最低速、brake 長度）；撳一行 → 跳去 `brake_start_m − 150`（lift／flat 用 `entry_m − 150`），0.5× 播到 `exit_m` 後自動暫停
- 下：播放列 ▶／⏸、0.25× 0.5× 1× 2×、可拖 timeline、時間
- 手機（< 768 px）：上下排 3D → 曲線 → 播放列；彎道清單收入「彎道」掣（覆蓋層）

### 鍵盤

`Space` 播放／暫停 · `←`/`→` ±2 s · `1`/`2`/`3` T-cam／Chase／Top

### 記住設定

localStorage（try/catch 包住）：view（遙測／彎道）、鏡頭。

## 6. 出錯處理（前端）

- `/pro_lap` 404 → 顯示「呢個車手／賽道冇數據」+ `available` 組合做可撳 chip
- 冇 WebGL → 3D 區域顯示提示，曲線同彎道清單照用
- FastF1 首次下載可能 > 30 s → loading 狀態（文字講明「首次下載中」）
- Three.js CDN 載入失敗 → 同冇 WebGL 一樣處理

## 7. 測試

`tests/test_pro_lap.py`（用 FastAPI TestClient + `2025_nor_q_bahrain.pkl`，唔需要網絡）：

1. `points` 每欄長度一致，> 2,000 點
2. 單位係米：XY 路徑長度同 `meta.length_m` 相差 < 5%
3. `t` 嚴格遞增；`t[-1]` 同 `meta.lap_time` 相差 < 0.5 s
4. 每個 `action == "brake"` 嘅彎：`brake_start_m < apex_m`
5. 未知 track → 404

前端：手動瀏覽器測試 + Playwright 截圖驗證（3 個鏡頭、2 個 view、手機闊度）。

**Notebook 豁免**：項目規則要求新分析加 notebook；呢個係呈現層，冇新分析算法，使用者已同意豁免。

## 8. 已知限制

- steering 係推算，唔係真實數據
- 職業車手 brake 只有開／關，冇力度 → brake 柱只會全滿或者空
- 賽道闊度固定 14 m；彎位 kerb、run-off 冇
- FastF1 GPS 原始約 4 Hz，平滑後彎位線條可能同真實 racing line 有幾米誤差
