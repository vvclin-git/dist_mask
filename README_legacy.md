# 距離分 bin 取樣與多邊形標註工具 v0.1

PowerShell 在專案資料夾執行：

```powershell
uv sync --locked
uv run --locked distance-annotator
```

使用 Python 3.12，Windows 需 tkinter（一般 Python 安裝附帶）。此工具獨立於 OCR 工具，不需要 PaddleOCR、模型或重新 optimize。

## 使用流程

1. 選擇上一工具產出的 `*_frames.csv` / `*_reviewed.csv`，並選擇對應原始影片。一次一支影片。
2. 距離欄位選 `buoy` 或 `ships`。設定 bin 寬度、每 bin 取樣數、最小 frame 間隔、seed。工具僅使用 accepted / manual_accepted 距離；依 distance_m 重分 bin，不依賴 CSV 原本 bin 寬度。
3. 設定標註 ROI `[x,y,width,height]`。這是物件影像範圍，與 OCR ROI 不同。預設 0,0,960,711 僅是起始範例，請依影片調整。可讀取影片後「手動畫 ROI」。ROI 必須在影像內。
4. 按「讀取並取樣」。取樣是固定 seed 隨機取樣；同影片跨 bin 也遵守最小間隔。若數量不足，保留可抽出的數量，不強制放寬間隔。抽樣完成後會自動依距離由小到大、同距離下依 frame 編號由小到大排序。左側顯示取樣清單；可選「匯出取樣截圖」產生原尺寸 PNG / samples.csv。
5. 選取 frame 後左鍵依序加入多邊形頂點，右鍵完成；可建立多個物件。撤銷可移除最後頂點或物件。按保存確認，或確認沒有物件。第一版為多邊形標註，可拖曳既有頂點微調；尚未包含筆刷與類別管理。
6. 左右方向鍵 / 按鈕切前後 frame；步長可調。也可輸入指定 frame 跳轉。鄰幀也能保存標註；若該 frame 距離沒有可靠值，距離及 bin 記錄為 null，不臆測補值。
7. 保存後目前 frame 成為參考。可用 Tab 在目前 / 參考影像切換，或選 overlay 並調透明度。滾輪縮放，中鍵拖曳；所有標註對應目前 frame 原始裁切座標。reference 模式禁止新增頂點。
8. 切到下一 frame 後按「預測目前物件位置」：使用已確認參考多邊形周圍影像，在局部範圍 template matching，平移全部物件形成橘色草稿。分數未達門檻不提供草稿。草稿需人工確認保存；不會自動更新 GT。
9. 「保存工作階段」保存設定與固定取樣清單；之後開啟 session.json 繼續。CSV 與影片須留在原路徑，若移動需調整 session 設定。每個 frame 標註保存後可再次修改覆寫，不覆寫來源影片 / 距離 CSV。
10. 「匯出標註索引」收集所有已確認 frame 至 annotations.json。

## 輸出

- `session.json`：設定、取樣清單與目前 frame。
- `samples/`：抽樣影像和清單。
- `images/`：已確認標註 frame 的原始 ROI。
- `masks/`：與 ROI 同尺寸的二值 mask，目標 255 / 背景 0；多物件取聯集。
- `annotated/`：綠色半透明 mask 預覽。
- `annotations/`：逐 frame JSON，包含影片、frame、距離、bin、ROI、每物件 polygon / bbox（裁切座標及原影片座標）、pixel area、聯集 pixel area、人工確認狀態。
- `annotations.json`：手動匯出的總索引。

## 限制

Template matching 只平移位置，不修正尺度、旋轉或輪廓。所有物件共用平移，獨立運動的多物件須人工調整頂點或清除重畫。半透明疊加不進行影像對齊。前後 frame 解碼有 16 張記憶體快取；首次跳轉速度依影片編碼與硬體。批次匯出在主介面執行，大量截圖時可能暫時無回應。

取樣間隔減少重複影像，並不能消除航次相關性；模型評估仍需依影片 / 航次切分。固定 gap 隨機貪婪取樣不保證最大可取樣數，且尚未分開接近 / 遠離航段。
