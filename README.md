# Distance Frame Annotator v2.1 — Pixel GT

距離分 bin 取樣與 Pixel Mask Annotator v3 整合版。一次處理一支影片；GT 為單一類別的聯集 mask，不保存獨立物件 ID。舊版說明另存為 README_legacy.md。

## 啟動

解壓縮至新資料夾，雙擊 `run.bat`；需先安裝 uv。或執行：

```powershell
uv sync --locked
uv run --locked distance-annotator
```

Python 3.12，需 tkinter（一般 Windows Python 安裝附帶），不需要 PaddleOCR。

## 工作流程

1. 選距離 CSV、對應影片及輸出資料夾，只使用 accepted / manual_accepted 距離列。
2. 設距離欄位 buoy / ships、bin、每 bin 數量、最小間隔與 seed。
3. 設**裁切 ROI** `[x,y,width,height]`，預設 `0,0,960,711` 是範例。按「載入／套用並取樣」，亦可手動畫裁切 ROI。
4. 固定 seed 隨機選取，再依距離、frame 升冪排列。最小間隔跨 bin 生效，數量不足時不放寬條件。
5. 畫**分析多邊形 ROI**，Enter 完成。這是色彩分析範圍，不是 GT，不改變裁切 ROI。
6. Analyze colors → 選色群 → Generate Candidate → Toggle Edit / Keep Component 修正 → Add Candidate to GT。
7. GT Brush / Eraser 微調，也可直接筆刷標記，無需分群。
8. Ctrl+S 保存非空 GT；無物件時按「確認無物件」。Candidate 必須先加入 GT 或清除才能保存。
9. 切回 frame 自動載入已保存 GT，未保存修改可選保存／捨棄／取消。

Candidate 加入 GT 是聯集，不刪除 GT；移除既有 GT 使用 GT Eraser。內部 mask 0/1，PNG 0/255；面積以非零數計算。

## 操作

| 操作 | 按鍵 |
|---|---|
| 前／後鄰幀 | Left / Right，依步長 |
| 前／後取樣 | Page Up / Page Down |
| 完成分析 ROI | Enter |
| 保存 GT | Ctrl+S |
| 移動整個 Candidate | M 或「移動 Candidate」按鈕，從候選像素內左鍵拖曳 |
| Undo / Redo | Ctrl+Z / Ctrl+Y |
| Pan | 右鍵、中鍵拖曳、Space + 左鍵 |
| 縮放 | 滾輪，以游標為中心 |
| ROI / Candidate 編修 / 連通區塊 | P / A / C |
| GT 筆刷 / 橡皮擦 | B / E |
| 清除分析 ROI | Esc，保留 Candidate / GT |
| 完整檢視 | Fit |

輸入框有焦點時停用標註／導覽快捷鍵。取樣按距離排序，不代表時間順序；切鄰幀保留取樣游標。換 frame 保留縮放和平移；裁切 ROI 或來源改變才 Fit。

新參數須按「載入／套用並取樣」。尚未套用時仍可保存目前舊設定的 GT；session 只保存已套用設定。

## 參考與傳播

「帶入最近保存 GT」將本工作階段最近一次成功保存的 GT 原位置複製到目前 Candidate，並設為參考；不需要 template matching。最近保存指操作順序，不是影片前一幀。來源 frame 顯示在狀態列，session 重開會恢復最近保存來源。

帶入後自動啟用移動工具：從候選像素內左鍵拖曳，影像與目前 GT 保持固定；按 A 回到增減像素編修。拖曳以整數像素平移，限制在裁切 ROI 內，不裁掉候選；每次拖曳可用一次 Undo／Redo 回復。接受為 GT 後不能用此工具移動 GT。

更換影片、距離欄位或裁切 ROI 時清除最近保存紀錄。最近保存為無物件時會提示沒有可帶入的 mask；帶入前若有 Candidate 則確認替換。既有 GT 不被覆寫。

保存後按「設目前確認 GT 為參考」，取得獨立副本。current / reference / overlay 與透明度可切換。reference 禁止編輯；overlay 編輯目前 frame，色彩分析始終使用目前原影像。

傳播使用局部 template matching、整數平移，結果只放入 Candidate。最近傳播嘗試記錄參考 frame、分數與位移，不代表 GT 未經修正。低分、空 GT、低紋理或越界時拒絕，替換既有 Candidate 前需確認。

相鄰幀通常更適合傳播；所有物件共同平移，不處理尺度、旋轉、形變或遮擋，須人工檢查。

## 保存與搬移

- `session.json`：已套用參數、固定取樣、目前 frame、游標、viewport；不保存未接受 Candidate／Undo。
- `annotation_index.json`：每個標註最新有效版本的相對路徑。
- `records/<source+class+frame+ROI>/<version>/`：image.png、mask.png、annotated.png、annotation.json。
- `samples/`：裁切影像與依距離排序的 samples.csv。
- `annotations.json`：手動匯出的總索引。annotation_file 相對於輸出資料夾；mask/image 路徑相對於 annotation JSON。

每次保存建立新版本，完成後才更新索引。舊版本保留，未索引版本不取代有效 GT；目前不自動清理歷史。

來源識別採大小及首尾資料的快速指紋，支援搬移，不是完整內容雜湊。session 記錄相對與原路徑，來源遺失可重新指定；指紋不符會拒絕。

舊 annotations/ JSON、masks/ PNG 依影片檔名、frame、ROI、欄位查找。優先 mask，缺少時由 polygon 重建；尺寸／數量不符會提示。舊檔不覆寫；沒有舊來源指紋，須人工確認影片。

重載只恢復 GT；分析、Candidate、Undo 不跨 frame。先前分析資訊保留於歷史 JSON，新版 supersedes 可追溯前版。

## 驗證與限制

```powershell
uv run --locked pytest -q
```

核心測試涵蓋保存重載、失敗保護、搬移、舊資料、負樣本、欄位隔離、取樣游標、viewport、參考唯讀與平移。環境無圖形顯示，Windows GUI 尚未實機驗證。

解碼與批次匯出仍在主執行緒，匯出前提示與顯示忙碌狀態，大量截圖暫停回應。色彩分析背景執行，檢查影像／ROI 版本。

缺乏可靠距離的鄰幀仍可標記，距離／bin 記 null。取樣間隔不消除航次相關性，訓練／驗證仍須按航次切分。

## 匯入 Track Lab 距離 CSV

1. 在「距離 CSV」選擇 Track Lab 匯出的目標距離–影格表。
2. 在「影片」選擇原始影片，檔名須與 CSV 的 `video_filename` 一致。
3. 在「距離欄位」填寫標註目標名稱（例如 `ships`），再按「載入／套用並取樣」。

程式以 `frame_number` 直接定位影格（從 0 開始），並以 `distance_m` 分 bin。
同一張表若包含多支影片，會依所選影片的檔名取用配對列。每支影片的有效列
須對應單一 AIS 目標。缺少有效距離或影格編號的列會略過。
Track Lab 的「距離欄位」用於標記輸出的 GT 類別。OCR 表仍依原有的 ROI 名稱與接受狀態篩選。

## 預設值、距離工作區與 seed

- 「保存為預設值」會將距離欄位、bin 大小、每 bin 張數、最小影格間隔、seed 和裁切 ROI 寫入使用者家目錄下的 `.dist_mask/defaults.json`，下次啟動自動載入。
- 距離工作區以公尺為單位，包含上下限，可任一側留空。按「載入／套用並取樣」後，只會從範圍內有有效距離的影格取樣，沿用原有 bin 邊界。手動跳轉和步進也受範圍限制。
- 工作區會隨工作階段保存。批次匯出的 `samples.csv` 只列出本次取樣，先前匯出的 PNG 和已保存的 GT 仍保留。
- 「產生新 seed」會更新 seed 欄位，按「載入／套用並取樣」後才套用。
