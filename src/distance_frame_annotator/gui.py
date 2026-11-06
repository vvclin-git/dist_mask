import csv,json,math
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import cv2
import numpy as np
from PIL import Image,ImageTk
from .core import load_rows,sample_rows,VideoReader,validate_roi,propagate,annotation_data,save_annotation,key_for

class App:
    def __init__(self,root):
        self.root=root;root.title('距離分 bin 取樣與多邊形標註');root.geometry('1400x950')
        self.rows=[];self.samples=[];self.reader=None;self.video_path=None;self.current=None;self.polygons=[];self.points=[];self.draft=False;self.dirty=False;self.reference=None;self.scale=1;self.zoom=1;self.offset=(0,0);self.drag_vertex=None
        self.v={k:tk.StringVar(value=v) for k,v in {'csv':'','video':'','out':'annotations_output','field':'buoy','width':'10','count':'3','gap':'30','seed':'42','roi':'0,0,960,711','step':'1','radius':'40','score':'0.75','frame':'0'}.items()}
        top=ttk.Frame(root);top.pack(fill='x')
        for i,(key,label,kind) in enumerate([('csv','距離 CSV','file'),('video','來源影片','video'),('out','輸出資料夾','dir')]):
            ttk.Label(top,text=label).grid(row=i,column=0);ttk.Entry(top,textvariable=self.v[key],width=100).grid(row=i,column=1,columnspan=6,sticky='ew');ttk.Button(top,text='選擇',command=lambda k=key,t=kind:self.pick(k,t)).grid(row=i,column=7)
        settings=ttk.Frame(root);settings.pack(fill='x')
        for k,label in [('field','距離欄位'),('width','bin(m)'),('count','每 bin 數'),('gap','最小間隔(frames)'),('seed','seed'),('roi','標註 ROI x,y,w,h')]:
            ttk.Label(settings,text=label).pack(side='left');ttk.Entry(settings,textvariable=self.v[k],width=22 if k=='roi' else 7).pack(side='left')
        bar=ttk.Frame(root);bar.pack(fill='x')
        for text,cmd in [('讀取並取樣',self.sample),('開啟工作階段',self.open_session),('保存工作階段',self.save_session),('手動畫 ROI',self.pick_roi),('匯出取樣截圖',self.export_samples),('匯出標註索引',self.export_index)]:ttk.Button(bar,text=text,command=lambda c=cmd:self.guard(c)).pack(side='left')
        body=ttk.Panedwindow(root,orient='horizontal');body.pack(fill='both',expand=True);left=ttk.Frame(body);right=ttk.Frame(body);body.add(left,weight=1);body.add(right,weight=5)
        self.list=ttk.Treeview(left,columns=('frame','distance','bin'),show='headings',selectmode='browse')
        for k in ['frame','distance','bin']:self.list.heading(k,text=k);self.list.column(k,width=85)
        self.list.pack(fill='both',expand=True);self.list.bind('<<TreeviewSelect>>',self.choose)
        self.canvas=tk.Canvas(right,bg='#222');self.canvas.pack(fill='both',expand=True);self.canvas.bind('<Configure>',lambda e:self.render());self.canvas.bind('<Button-1>',self.click);self.canvas.bind('<B1-Motion>',self.drag);self.canvas.bind('<ButtonRelease-1>',lambda e:setattr(self,'drag_vertex',None));self.canvas.bind('<Button-3>',lambda e:self.finish());self.canvas.bind('<MouseWheel>',self.wheel);self.canvas.bind('<ButtonPress-2>',lambda e:self.canvas.scan_mark(e.x,e.y));self.canvas.bind('<B2-Motion>',lambda e:self.canvas.scan_dragto(e.x,e.y,gain=1))
        nav=ttk.Frame(root);nav.pack(fill='x')
        for text,cmd in [('◀ 前 frame',lambda:self.move(-1)),('後 frame ▶',lambda:self.move(1)),('前取樣',lambda:self.next_sample(-1)),('後取樣',lambda:self.next_sample(1))]:ttk.Button(nav,text=text,command=lambda c=cmd:self.guard(c)).pack(side='left')
        for k,label in [('step','步長'),('frame','跳至 frame')]:ttk.Label(nav,text=label).pack(side='left');ttk.Entry(nav,textvariable=self.v[k],width=8).pack(side='left')
        ttk.Button(nav,text='跳轉',command=lambda:self.guard(lambda:self.show(int(self.v['frame'].get())))).pack(side='left');ttk.Button(nav,text='設目前為參考',command=self.set_reference).pack(side='left')
        self.mode=tk.StringVar(value='current');ttk.Combobox(nav,textvariable=self.mode,values=['current','reference','overlay'],state='readonly',width=10).pack(side='left');self.mode.trace_add('write',lambda *a:self.render());self.alpha=tk.DoubleVar(value=.5);ttk.Scale(nav,from_=0,to=1,variable=self.alpha,command=lambda v:self.render()).pack(side='left')
        tools=ttk.Frame(root);tools.pack(fill='x')
        for text,cmd in [('完成多邊形 (右鍵)',self.finish),('撤銷',self.undo),('清除',self.clear),('預測目前物件位置',self.predict),('保存 / 確認草稿',self.save),('確認沒有物件',lambda:self.save(True))]:ttk.Button(tools,text=text,command=lambda c=cmd:self.guard(c)).pack(side='left')
        for k,label in [('radius','搜尋半徑'),('score','最低分數')]:ttk.Label(tools,text=label).pack(side='left');ttk.Entry(tools,textvariable=self.v[k],width=6).pack(side='left')
        self.status=tk.StringVar(value='左鍵加頂點；右鍵完成。←/→ 切 frame；Tab 切參考；Enter 保存；滾輪縮放；中鍵拖曳。');ttk.Label(root,textvariable=self.status).pack(fill='x')
        root.bind('<Left>',lambda e:self.shortcut(lambda:self.move(-1)));root.bind('<Right>',lambda e:self.shortcut(lambda:self.move(1)));root.bind('<Tab>',self.toggle);root.bind('<Return>',lambda e:self.shortcut(self.save));root.bind('<BackSpace>',lambda e:self.shortcut(self.undo));root.protocol('WM_DELETE_WINDOW',self.close)
    def guard(self,cmd):
        try:cmd()
        except Exception as e:messagebox.showerror('Error',str(e))
    def shortcut(self,cmd):
        if isinstance(self.root.focus_get(),(tk.Entry,ttk.Entry,ttk.Combobox)):return
        self.guard(cmd)
    def pick(self,key,kind):
        p=filedialog.askdirectory() if kind=='dir' else filedialog.askopenfilename(filetypes=[('CSV','*.csv'),('All','*')]) if kind=='file' else filedialog.askopenfilename(filetypes=[('Video','*.mp4 *.avi *.mov *.mkv'),('All','*')])
        if p:self.v[key].set(p)
    def roi(self):
        r=[int(x.strip()) for x in self.v['roi'].get().split(',')]
        if len(r)!=4:raise ValueError('ROI must contain x,y,width,height')
        return r
    def discard(self):return not self.dirty or messagebox.askyesno('未保存標註','捨棄目前尚未保存的修改？')
    def sample(self):
        if not self.discard():return
        rows=load_rows(self.v['csv'].get(),self.v['field'].get(),self.v['video'].get());names={r['video'] for r in rows}
        if len(names)!=1:raise ValueError('第一版一次處理一支影片；請提供單一影片的 CSV')
        video=Path(self.v['video'].get() or next(iter(names)))
        if not video.is_file():raise ValueError('請選擇對應來源影片')
        reader=VideoReader(video)
        try:validate_roi(self.roi(),reader.get(0))
        except Exception:reader.close();raise
        if self.reader:self.reader.close()
        self.reader=reader;self.reference=None;self.video_path=video.resolve();self.v['video'].set(str(self.video_path));self.rows=rows;self.lookup={r['frame_index']:r for r in rows}
        self.samples=sample_rows(rows,float(self.v['width'].get()),int(self.v['count'].get()),int(self.v['gap'].get()),int(self.v['seed'].get()));self.dirty=False
        self.fill_list();self.save_session()
        if self.samples:self.show(self.samples[0]['frame_index'])
        self.status.set(f'已取樣 {len(self.samples)} frames；間隔不足的 bin 可能少於要求數量。')
    def fill_list(self):
        self.list.delete(*self.list.get_children())
        for i,r in enumerate(self.samples):self.list.insert('','end',iid=str(i),values=(r['frame_index'],r['distance_m'],r['bin_lower_m']))
    def choose(self,e=None):
        ids=self.list.selection()
        if ids:self.guard(lambda:self.show(self.samples[int(ids[0])]['frame_index']))
    def row(self,index):
        r=self.lookup.get(index)
        if r is None:return {'frame_index':index,'roi':self.v['field'].get(),'distance_m':None,'timestamp_s':'','bin_lower_m':None}
        return {**r,'bin_lower_m':math.floor(r['distance_m']/float(self.v['width'].get()))*float(self.v['width'].get())}
    def show(self,index):
        if self.reader is None:raise ValueError('請先讀取 CSV 或工作階段')
        if not self.discard():return
        frame=self.reader.get(index);crop=validate_roi(self.roi(),frame)
        if hasattr(self,'used_roi') and self.used_roi!=self.roi():self.reference=None
        self.current=index;self.used_roi=self.roi();self.crop=crop;self.points=[];self.polygons=[];self.draft=False;self.dirty=False;self.v['frame'].set(str(index))
        p=Path(self.v['out'].get())/'annotations'/(key_for(self.video_path,index,self.roi())+'.json')
        if p.exists():
            data=json.loads(p.read_text(encoding='utf-8'));self.polygons=[o['polygon_roi'] for o in data['objects']]
        self.status.set(f'Frame {index} | 距離 {self.row(index)["distance_m"]} m | 已保存: {p.exists()}');self.render()
    def move(self,d):
        if self.current is not None:self.show(self.current+d*int(self.v['step'].get()))
    def next_sample(self,d):
        if not self.samples:return
        indices=[r['frame_index'] for r in self.samples]
        i=min(range(len(indices)),key=lambda i:abs(indices[i]-(self.current or 0)));i=max(0,min(len(indices)-1,i+d));self.list.selection_set(str(i));self.list.see(str(i))
    def pick_roi(self):
        if self.reader is None:raise ValueError('請先讀取影片')
        if not self.discard():return
        f=self.reader.get(self.current or 0);r=cv2.selectROI('Select annotation ROI - ENTER',f,False,False);cv2.destroyAllWindows()
        if r[2] and r[3]:self.v['roi'].set(','.join(map(str,r)));self.reference=None;self.dirty=False;self.show(self.current or 0)
    def set_reference(self):
        if self.current is None:return
        self.reference=(self.current,self.crop.copy(),json.loads(json.dumps(self.polygons)),not self.dirty and not self.draft);self.status.set(f'參考 frame: {self.current}');self.render()
    def toggle(self,e=None):
        if isinstance(self.root.focus_get(),(tk.Entry,ttk.Entry,ttk.Combobox)):return
        self.mode.set('reference' if self.mode.get()=='current' else 'current');return 'break'
    def wheel(self,e):self.zoom=max(.25,min(8,self.zoom*(1.2 if e.delta>0 else 1/1.2)));self.render()
    def render(self):
        if self.current is None:return
        im=self.crop.copy();mode=self.mode.get()
        if self.reference and self.reference[1].shape==im.shape:
            if mode=='reference':im=self.reference[1].copy()
            elif mode=='overlay':im=cv2.addWeighted(im,1-self.alpha.get(),self.reference[1],self.alpha.get(),0)
        cw,ch=max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height());h,w=im.shape[:2];self.scale=min(cw/w,ch/h)*self.zoom
        dw,dh=max(1,round(w*self.scale)),max(1,round(h*self.scale));self.offset=(max(0,(cw-dw)//2),max(0,(ch-dh)//2));self.photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im,cv2.COLOR_BGR2RGB)).resize((dw,dh)))
        self.canvas.delete('all');ox,oy=self.offset;self.canvas.create_image(ox,oy,image=self.photo,anchor='nw');self.canvas.configure(scrollregion=(0,0,max(cw,dw),max(ch,dh)))
        polys=self.reference[2] if mode=='reference' and self.reference else self.polygons
        for poly in polys:
            coords=[v for x,y in poly for v in (ox+x*self.scale,oy+y*self.scale)]
            if len(coords)>=6:self.canvas.create_polygon(*coords,fill='',outline='#ffb000' if self.draft else '#00ff80',width=2)
        if mode!='reference' and self.points:
            coords=[v for x,y in self.points for v in (ox+x*self.scale,oy+y*self.scale)]
            if len(coords)>=4:self.canvas.create_line(*coords,fill='cyan',width=2)
            for x,y in self.points:self.canvas.create_oval(ox+x*self.scale-3,oy+y*self.scale-3,ox+x*self.scale+3,oy+y*self.scale+3,fill='cyan')
    def click(self,e):
        if self.current is None or self.mode.get()=='reference':return
        ox,oy=self.offset;x=round((self.canvas.canvasx(e.x)-ox)/self.scale);y=round((self.canvas.canvasy(e.y)-oy)/self.scale)
        if not self.points:
            for i,poly in enumerate(self.polygons):
                for j,(px,py) in enumerate(poly):
                    if (px-x)**2+(py-y)**2 <= (8/self.scale)**2:
                        self.drag_vertex=(i,j);return
        if 0<=x<self.crop.shape[1] and 0<=y<self.crop.shape[0]:self.points.append([x,y]);self.dirty=True;self.render()
    def drag(self,e):
        if self.drag_vertex is None or self.mode.get()=='reference':return
        ox,oy=self.offset;x=round((self.canvas.canvasx(e.x)-ox)/self.scale);y=round((self.canvas.canvasy(e.y)-oy)/self.scale)
        i,j=self.drag_vertex;self.polygons[i][j]=[max(0,min(self.crop.shape[1]-1,x)),max(0,min(self.crop.shape[0]-1,y))];self.dirty=True;self.render()
    def finish(self):
        if len(self.points)>=3:self.polygons.append(self.points);self.points=[];self.dirty=True;self.render()
    def undo(self):
        if self.points:self.points.pop()
        elif self.polygons:self.polygons.pop()
        self.dirty=True;self.render()
    def clear(self):self.points=[];self.polygons=[];self.draft=False;self.dirty=True;self.render()
    def predict(self):
        if not self.reference or not self.reference[3]:raise ValueError('請先保存參考 frame 的標註，再設為參考')
        if not self.discard():return
        radius=int(self.v['radius'].get());score=float(self.v['score'].get())
        if radius<1 or not 0<=score<=1:raise ValueError('Invalid matching parameters')
        polys,value=propagate(self.reference[1],self.crop,self.reference[2],radius,score)
        if not polys:raise ValueError(f'未達門檻，matching score={value:.3f}')
        self.polygons=polys;self.points=[];self.draft=True;self.dirty=True;self.mode.set('current');self.status.set(f'位置預測草稿 score={value:.3f}；請檢查輪廓後保存確認');self.render()
    def save(self,negative=False):
        if self.current is None:return
        if self.roi()!=self.used_roi:raise ValueError('ROI 已變更，請重新開啟目前 frame 再標註')
        if self.points:raise ValueError('請先完成多邊形或撤銷未完成頂點')
        if negative and self.polygons:raise ValueError('請先清除多邊形，再確認沒有物件')
        record,mask=annotation_data(self.row(self.current),self.video_path,self.roi(),self.polygons,self.crop,negative)
        record['bin_width_m']=float(self.v['width'].get());record['proposal_method']='template_matching' if self.draft else 'manual';save_annotation(self.v['out'].get(),record,self.crop,mask);self.dirty=False;self.draft=False;self.set_reference();self.save_session();self.status.set(f'Frame {self.current} 已保存人工確認標註');self.render()
    def save_session(self):
        out=Path(self.v['out'].get());out.mkdir(parents=True,exist_ok=True);data={'settings':{k:v.get() for k,v in self.v.items()},'samples':self.samples,'current_frame':self.current}
        p=out/'session.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
    def open_session(self):
        if not self.discard():return
        p=filedialog.askopenfilename(filetypes=[('Session','*.json')])
        if not p:return
        data=json.loads(Path(p).read_text(encoding='utf-8'))
        for k,value in data['settings'].items():
            if k in self.v:self.v[k].set(value)
        self.rows=load_rows(self.v['csv'].get(),self.v['field'].get(),self.v['video'].get());self.lookup={r['frame_index']:r for r in self.rows};reader=VideoReader(self.v['video'].get())
        if self.reader:self.reader.close()
        self.reader=reader;self.video_path=Path(self.v['video'].get()).resolve();self.samples=data['samples'];self.dirty=False;self.fill_list();self.show(data.get('current_frame') or self.samples[0]['frame_index'])
    def export_samples(self):
        if self.reader is None:raise ValueError('請先取樣')
        out=Path(self.v['out'].get())/'samples';out.mkdir(parents=True,exist_ok=True);records=[]
        for row in self.samples:
            crop=validate_roi(self.roi(),self.reader.get(row['frame_index']));p=out/(key_for(self.video_path,row['frame_index'],self.roi())+'.png');cv2.imencode('.png',crop)[1].tofile(str(p));records.append({**row,'image':str(p.resolve()),'crop_roi':json.dumps(self.roi())})
        if records:
            with (out/'samples.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
        self.status.set(f'已匯出 {len(records)} 張原尺寸取樣 ROI')
    def export_index(self):
        out=Path(self.v['out'].get());records=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((out/'annotations').glob('*.json'))];(out/'annotations.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8');self.status.set(f'已匯出 {len(records)} 個已確認 frame 的 annotations.json')
    def close(self):
        if not self.discard():return
        if self.reader:self.reader.close()
        self.root.destroy()

def main():
    root=tk.Tk();App(root);root.mainloop()

if __name__=='__main__':main()
