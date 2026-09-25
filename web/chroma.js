/* Conservative chroma routing and local color projection, matching chroma.py.
   Nearest-color propagation uses grid distance; Python uses Euclidean distance. */
export function inspect(data,w,h){
 const samples=[];for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(x<4||y<4||x>=w-4||y>=h-4){const i=(y*w+x)*4;samples.push([data[i]/255,data[i+1]/255,data[i+2]/255]);}
 const green=samples.filter(p=>p[1]-Math.max(p[0],p[2])>.3&&p[1]>.45),fraction=green.length/samples.length;
 const median=a=>{a.sort((x,y)=>x-y);return a.length%2?a[(a.length-1)/2]:(a[a.length/2-1]+a[a.length/2])/2;};
 const key=green.length?[0,1,2].map(c=>median(green.map(p=>p[c]))):[0,1,0];const deviations=green.map(p=>Math.max(...p.map((v,c)=>Math.abs(v-key[c])))).sort((a,b)=>a-b);const spread=deviations.length?deviations[Math.floor((deviations.length-1)*.95)]:1;
 return {preserve:fraction<=.15,uniform:fraction>.85&&spread<.16,key,fraction,spread};
}
function erode(mask,w,h,r){const out=new Uint8Array(mask.length);for(let y=0;y<h;y++)for(let x=0;x<w;x++){let v=1;for(let dy=-r;dy<=r&&v;dy++)for(let dx=-r;dx<=r;dx++)if(!mask[Math.max(0,Math.min(h-1,y+dy))*w+Math.max(0,Math.min(w-1,x+dx))]){v=0;break;}out[y*w+x]=v;}return out;}
function nearest(mask,w,h){const n=mask.length,labels=new Int32Array(n).fill(-1),dist=new Int32Array(n),queue=new Int32Array(n);let head=0,tail=0;for(let i=0;i<n;i++)if(mask[i]){labels[i]=i;queue[tail++]=i;}while(head<tail){const i=queue[head++],x=i%w,y=(i/w)|0;for(const j of [x?i-1:-1,x<w-1?i+1:-1,y?i-w:-1,y<h-1?i+w:-1])if(j>=0&&labels[j]<0){labels[j]=labels[i];dist[j]=dist[i]+1;queue[tail++]=j;}}return {labels,dist};}
export function refine(data,w,h){
 const n=w*h,fg=new Uint8Array(n),bg=new Uint8Array(n);for(let j=0;j<n;j++){const i=4*j,ex=(data[i+1]-Math.max(data[i],data[i+2]))/255;fg[j]=ex<.025;bg[j]=ex>.35&&data[i+1]/255>.5;}
 const out=new Uint8ClampedArray(data.length);if(!fg.some(Boolean))return out;if(!bg.some(Boolean)){out.set(data);return out;}
 let inside=erode(fg,w,h,2);if(!inside.some(Boolean))inside=fg;let outside=erode(bg,w,h,1);if(!outside.some(Boolean))outside=bg;
 const fi=nearest(inside,w,h).labels,bi=nearest(outside,w,h).labels,distance=nearest(fg,w,h).dist,a=new Float32Array(n);
 for(let j=0;j<n;j++){let dot=0,den=0;for(let c=0;c<3;c++){const f=data[fi[j]*4+c]/255,k=data[bi[j]*4+c]/255,v=f-k;dot+=(data[j*4+c]/255-k)*v;den+=v*v;}let alpha=Math.max(0,Math.min(1,dot/Math.max(den,1e-6)));if(fg[j])alpha=1;if(bg[j]&&distance[j]>8)alpha=0;alpha=alpha<.01?0:alpha>.99?1:alpha;a[j]=alpha;for(let c=0;c<3;c++)out[j*4+c]=alpha===0?0:data[(alpha<1?fi[j]:j)*4+c];out[j*4+3]=Math.round(alpha*255);}
 for(let y=0;y<h;y++)for(let x=0;x<w;x++){const j=y*w+x;if(a[j]===0)continue;let edge=false;for(let dy=-3;dy<=3&&!edge;dy++)for(let dx=-3;dx<=3;dx++){if(a[Math.max(0,Math.min(h-1,y+dy))*w+Math.max(0,Math.min(w-1,x+dx))]<.99){edge=true;break;}}if(edge)out[j*4+1]=Math.min(out[j*4+1],Math.max(out[j*4],out[j*4+2]));}
 return out;
}
