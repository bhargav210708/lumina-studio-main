// Lumina Studio core: single-pass WebGL2 pipeline (fp16 intermediates), 3D LUT, halation, grain, scopes, exports.
const $=s=>document.querySelector(s), cv=$('#view');
const gl=cv.getContext('webgl2',{preserveDrawingBuffer:true});
if(!gl){document.body.innerHTML='WebGL2 required';throw 0}
gl.getExtension('EXT_color_buffer_float');

const VS=`#version 300 es
in vec2 p;out vec2 uv;void main(){uv=p*.5+.5;gl_Position=vec4(p,0,1);}`;
const FS=`#version 300 es
precision highp float;precision highp sampler3D;
in vec2 uv;out vec4 o;
uniform sampler2D uTex,uCrv;uniform vec3 hsl[8];uniform sampler2D uMask,uDepth;uniform float dyeC,dyeM,dyeY,smear,sang,flare,streakT,gpush,mel,hemo,dering,adjc,hsW,hsShift,hsSat,ptAmt,hsOn,hsHue,ptOn,zhud,zsel;uniform float zadj[11];uniform sampler2D uPT;uniform float gType,gAmt,gScale,gAng,gSoft,gWarm;uniform vec4 mA[4],mB[4],mC[4];uniform sampler2D uBrush;uniform float mShow;uniform float lab,negS,prtS,labView,ptR,ptG,ptB,dev,silver,scr,strk,dust,gwv;uniform float relOn,lx,ly,lint,lwarm,stretch,starmin,grad;uniform vec3 bgc[4];uniform float maskOn,subj,bg,skin,neg,mask;uniform float lutN,lutN2;uniform sampler3D uLut2;uniform vec3 lift,gam,gain;uniform sampler3D uLut;uniform vec2 uPx;
uniform float ev,con,hi,sh,temp,tint,sat,vig,halo,grain,mono,lutOn,lutMix,bake,t,orig,zone,clar,cpl,aud,dens,gnd,gndY,barrel,ca,streak,lutOn2,lutMix2,zeb,fc,frame,ia;
vec3 toLin(vec3 c){return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(.04045,c));}
vec3 toSrgb(vec3 c){c=max(c,0.);return mix(c*12.92,1.055*pow(c,vec3(1./2.4))-.055,step(.0031308,c));}
float luma(vec3 c){return dot(c,vec3(.2126,.7152,.0722));}
float h(vec2 s){return fract(sin(dot(s,vec2(12.9898,78.233)))*43758.5453);}
vec3 rgb2hsv(vec3 c){vec4 K=vec4(0.,-1./3.,2./3.,-1.),p=mix(vec4(c.bg,K.wz),vec4(c.gb,K.xy),step(c.b,c.g)),q=mix(vec4(p.xyw,c.r),vec4(c.r,p.yzx),step(p.x,c.r));float d=q.x-min(q.w,q.y),e=1e-10;return vec3(abs(q.z+(q.w-q.y)/(6.*d+e)),d/(q.x+e),q.x);}
vec3 hsv2rgb(vec3 c){vec4 K=vec4(1.,2./3.,1./3.,3.);vec3 p=abs(fract(c.xxx+K.xyz)*6.-K.www);return c.z*mix(K.xxx,clamp(p-K.xxx,0.,1.),c.y);}
const float HC[8]=float[8](0.,.083,.167,.333,.5,.667,.75,.833);
vec3 tet(sampler3D uLut,float lutN,vec3 c){float n=lutN-1.;vec3 p=clamp(c,0.,1.)*n;vec3 i0=floor(min(p,n-.001));vec3 f=p-i0;ivec3 b=ivec3(i0);
 vec3 c000=texelFetch(uLut,b,0).rgb,c111=texelFetch(uLut,b+1,0).rgb,c100=texelFetch(uLut,b+ivec3(1,0,0),0).rgb,c010=texelFetch(uLut,b+ivec3(0,1,0),0).rgb,c001=texelFetch(uLut,b+ivec3(0,0,1),0).rgb,c110=texelFetch(uLut,b+ivec3(1,1,0),0).rgb,c101=texelFetch(uLut,b+ivec3(1,0,1),0).rgb,c011=texelFetch(uLut,b+ivec3(0,1,1),0).rgb;
 if(f.x>f.y){if(f.y>f.z)return c000+(c100-c000)*f.x+(c110-c100)*f.y+(c111-c110)*f.z;
  if(f.x>f.z)return c000+(c100-c000)*f.x+(c101-c100)*f.z+(c111-c101)*f.y;
  return c000+(c001-c000)*f.z+(c101-c001)*f.x+(c111-c101)*f.y;}
 if(f.z>f.y)return c000+(c001-c000)*f.z+(c011-c001)*f.y+(c111-c011)*f.x;
 if(f.z>f.x)return c000+(c010-c000)*f.y+(c011-c010)*f.z+(c111-c011)*f.x;
 return c000+(c010-c000)*f.y+(c110-c010)*f.x+(c111-c110)*f.z;}
float vn(vec2 p){vec2 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);return mix(mix(h(i),h(i+vec2(1,0)),f.x),mix(h(i+vec2(0,1)),h(i+vec2(1,1)),f.x),f.y);}
float goboM(){vec2 p=(uv-.5)*vec2(uPx.y/uPx.x,1.);float a=gAng*3.14159,ca=cos(a),sa=sin(a);p=vec2(ca*p.x-sa*p.y,sa*p.x+ca*p.y);float s=2.+gScale*10.,sf=.02+gSoft*.3,m=1.;
  if(gType<1.5)m=smoothstep(-sf,sf,sin(p.y*s*6.2832));
  else if(gType<2.5)m=smoothstep(.5-sf,.5+sf,vn(p*s*1.5)*.6+vn(p*s*3.)*.4);
  else if(gType<3.5){float e=.5-max(abs(fract(p.x*s*.5)-.5),abs(fract(p.y*s*.5)-.5));m=smoothstep(.02,.05+sf*.5,e);}
  else if(gType<4.5){vec2 q=p*s*2.,g=floor(q),f=fract(q);m=1.-smoothstep(.12,.3+sf,length(f-vec2(h(g),h(g+7.))*.6-.2));}
  else{vec2 q=p*s*1.2,g=floor(q),f=fract(q);m=1.-smoothstep(.15,.3+sf,length(f-.5+(vec2(h(g),h(g+5.))-.5)*.5));}
  return m;}
float maskOf(int i,vec3 c){vec4 A=mA[i],B=mB[i];float t=A.x,m=0.;if(t<.5)return 0.;
  if(t<1.5){vec2 p0=A.yz,d=vec2(A.w,B.x)-p0;m=1.-smoothstep(0.,1.,clamp(dot(uv-p0,d)/max(dot(d,d),1e-4),0.,1.));}
  else if(t<2.5){float d=length((uv-A.yz)/max(vec2(A.w,B.x),vec2(.01)));m=1.-smoothstep(1.-B.y,1.001,d);}
  else if(t<3.5){float L=pow(max(luma(c),0.),1./2.2);m=smoothstep(A.y-B.y,A.y,L)*(1.-smoothstep(A.w,A.w+B.y,L));}
  else if(t<4.5){vec3 h=rgb2hsv(toSrgb(max(c,vec3(0.))));float d=abs(h.x-A.y);d=min(d,1.-d);m=(1.-smoothstep(A.w,A.w+B.y,d))*smoothstep(.08,.2,h.y);}
  else m=texture(uBrush,uv)[i];
  return B.z>.5?1.-m:m;}
vec3 labFilm(vec3 e){
  if(negS>2.5)e=vec3(dot(e,vec3(.2126,.7152,.0722)));
  float k=(negS<.5?.36:negS<1.5?.40:negS<2.5?.46:.40)*(1.+dev*.25),dmx=negS<.5?1.9:negS<1.5?2.1:negS<2.5?2.2:2.0;
  vec3 dm=(negS>2.5?vec3(.12):vec3(.28,.62,.95))*mask;
  vec3 t=1./(1.+exp(-k*(log2(max(e,vec3(1e-4)))+2.47)));
  vec3 Dn=dmx*t;mat3 M=mat3(1.,.07,.02,.04,1.,.10,.01,.05,1.);Dn=M*Dn;
  if(labView>.5)return pow(vec3(10.),-(Dn+dm)*.45);
  float kp=prtS<.5?3.:2.7,dM=prtS<.5?3.8:3.6,dm0=prtS<.5?.06:.08;
  vec3 x=-Dn+vec3(ptR,ptG,ptB)*.025;
  vec3 Dp=dm0+(dM-dm0)/(1.+exp(-kp*(x+dmx*.5)));
  Dp+=silver*dot(Dp,vec3(.3333))*.5;
  return pow(vec3(10.),-Dp*.4);}
vec3 grade(vec3 c){
  c*=exp2(ev);
  vec3 Dd=-log2(max(c,vec3(1e-4)))*(1.+vec3(dyeC,dyeM,dyeY)*.5);c=exp2(-Dd);
  float zz=clamp(5.+log2(max(luma(c),1e-4)/.18),0.,9.999);int zi=int(zz);c*=exp2(mix(zadj[zi],zadj[zi+1],smoothstep(0.,1.,fract(zz))));
  c*=vec3(1.+(temp+aud*.4)*.3,1.+tint*-.15,1.-(temp+aud*.4)*.3);
  float l=luma(c);
  c+=c*sh*.6*(1.-smoothstep(0.,.35,l))+c*hi*.5*smoothstep(.35,1.,l);
  c=max(c,0.);
  c=toSrgb(c);
  c=(c-.5)*(1.+con+aud*.3)+.5;                       // contrast in display space
  float y=luma(c);c=mix(vec3(y),c,1.+sat);
  vec3 hv=rgb2hsv(c);float dh=0.,ds=0.,dl=0.;
  for(int i=0;i<8;i++){float d=abs(hv.x-HC[i]);d=min(d,1.-d);float w=max(0.,1.-d/.14)*hv.y;dh+=w*hsl[i].x;ds+=w*hsl[i].y;dl+=w*hsl[i].z;}
  hv.x=fract(hv.x+dh*.08);hv.y=clamp(hv.y*(1.+ds),0.,1.);hv.z=clamp(hv.z*(1.+dl*.5),0.,1.);float sk=smoothstep(.5,.6,hv.x)*(1.-smoothstep(.72,.8,hv.x));hv.y=clamp(hv.y*(1.+cpl*.5*sk),0.,1.);hv.z*=1.-cpl*.3*sk*hv.y;if(hsOn>.5){float dd=abs(hv.x-hsHue);dd=min(dd,1.-dd);float w=(1.-smoothstep(0.,max(hsW,.001),dd))*smoothstep(.04,.2,hv.y);hv.x=fract(hv.x+hsShift*.25*w);hv.y=clamp(hv.y*(1.+hsSat*w),0.,1.);}
  c=hsv2rgb(hv);
  {float hx=hv.x>.9?hv.x-1.:hv.x,sw=smoothstep(-.01,.03,hx)*(1.-smoothstep(.09,.14,hx))*smoothstep(.12,.25,hv.y)*(1.-smoothstep(.65,.8,hv.y));
   if(bake<.5&&maskOn>.5)sw=max(sw*.25,texture(uMask,uv).g);
   vec3 OD=-log2(max(c,vec3(1e-3)));OD+=sw*(mel*.5*normalize(vec3(.3,.55,.78))+hemo*.5*normalize(vec3(.05,.8,.5)));c=exp2(-OD);}
  float chr=max(c.r,max(c.g,c.b))-min(c.r,min(c.g,c.b));c*=1.-dens*chr*.6;
  c=pow(max(c,0.),1./max(1.+gam*1.2,.3));c=c*(1.+gain*1.2)+lift*.3*(1.-c);
  c=mix(c,vec3(luma(c)),mono);
  c=clamp(c,0.,1.);c=vec3(texture(uCrv,vec2((c.r*255.+.5)/256.,.5)).r,texture(uCrv,vec2((c.g*255.+.5)/256.,.5)).g,texture(uCrv,vec2((c.b*255.+.5)/256.,.5)).b);
  return clamp(c,0.,1.);}
void main(){
  if(orig>.5&&bake<.5){o=vec4(texture(uTex,uv).rgb,1);return;}
  float bk=step(bake,.5);vec2 q0=uv-.5;vec2 u2=.5+q0*(1.+barrel*bk*dot(q0,q0)*2.)+bk*gwv*.004*vec2(sin(t*1.7),cos(t*2.3));float cs=ca*bk*.01;
  vec3 c=toLin(vec3(texture(uTex,.5+(u2-.5)*(1.+cs)).r,texture(uTex,u2).g,texture(uTex,.5+(u2-.5)*(1.-cs)).b));
  if(bake<.5){
    if(smear>0.){vec2 sd=vec2(cos(sang*1.5708),sin(sang*1.5708))*uPx*smear*40.;vec3 sm=c;for(int i=1;i<=6;i++){float f=float(i)/6.;sm+=toLin(texture(uTex,u2+sd*f).rgb)+toLin(texture(uTex,u2-sd*f).rgb);}c=sm/13.;}
    vec3 bl=(clar==0.&&dering==0.&&adjc==0.)?c:(toLin(texture(uTex,uv+uPx*vec2(5,0)).rgb)+toLin(texture(uTex,uv-uPx*vec2(5,0)).rgb)+toLin(texture(uTex,uv+uPx*vec2(0,5)).rgb)+toLin(texture(uTex,uv-uPx*vec2(0,5)).rgb))*.25;
    c+=(c-bl)*clar*1.5;c=min(c,mix(c,bl,dering));c+=(c-bl)*adjc*.5/(1.+abs(c-bl)*8.);c*=exp2(-gnd*smoothstep(-.2,.2,uv.y-.5-gndY));
    if(gType>.5&&gAmt>0.){float gm=goboM();c*=mix(1.-gAmt,1.+gAmt*.35,gm);c*=mix(vec3(1.),vec3(1.+gWarm*.3,1.,1.-gWarm*.3),gm*gAmt);}
    for(int i=0;i<4;i++){float mk=maskOf(i,c);if(mk>0.){vec4 C=mC[i];c*=exp2(C.x*mk);c=mix(vec3(luma(c)),c,1.+C.z*mk);c*=vec3(1.+C.w*mk*.3,1.,1.-C.w*mk*.3);c=max((c-.18)*(1.+C.y*mk)+.18,0.);}
      if(float(i)==mShow)c=mix(c,vec3(1.,.1,.2),maskOf(i,c)*.6);}
    if(maskOn>.5){vec3 m=texture(uMask,uv).rgb;c*=exp2(subj*m.r+bg*(1.-m.r));c*=vec3(1.+skin*.2*m.g,1.,1.-skin*.2*m.g);}
    if(relOn>.5){float d0=texture(uDepth,uv).r,dx=texture(uDepth,uv+vec2(uPx.x*3.,0)).r-texture(uDepth,uv-vec2(uPx.x*3.,0)).r,dy=texture(uDepth,uv+vec2(0,uPx.y*3.)).r-texture(uDepth,uv-vec2(0,uPx.y*3.)).r;
      vec3 nn=normalize(vec3(-dx*10.,-dy*10.,1.)),Ld=vec3(.5+lx,.5+ly,.35)-vec3(uv,d0*.5);float di=length(Ld);
      c+=c*max(dot(nn,Ld/di),0.)*lint*vec3(1.+lwarm*.4,1.,1.-lwarm*.4)/(1.+di*di*4.);}
    if(grad>0.)c=max(c-mix(mix(bgc[0],bgc[1],uv.x),mix(bgc[2],bgc[3],uv.x),uv.y)*grad,0.);
    if(starmin>0.){vec3 b2=(toLin(texture(uTex,uv+uPx*vec2(9,0)).rgb)+toLin(texture(uTex,uv-uPx*vec2(9,0)).rgb)+toLin(texture(uTex,uv+uPx*vec2(0,9)).rgb)+toLin(texture(uTex,uv-uPx*vec2(0,9)).rgb))*.25;c-=max(c-b2,0.)*starmin*.9;}
    if(stretch>0.){float k=1.+stretch*40.;c=asinh(c*k)/asinh(k);}}
  if(bake<.5){
    // halation: red-biased bloom from bright neighbours
    vec3 b=vec3(0);
    for(int i=0;i<(halo>0.?16:0);i++){float a=float(i)*.3927;vec2 d=vec2(cos(a),sin(a))*uPx*(6.+float(i%4)*6.);
      vec3 s=toLin(texture(uTex,uv+d).rgb);float dd=(6.+float(i%4)*6.)/10.;b+=max(s-.8,0.)/(1.+dd*dd);}
    c+=b/16.*halo*vec3(1.,.28,.08)*2.;
    if(flare>0.){vec3 gh=vec3(0);for(int i=0;i<3;i++){float k=-.35-float(i)*.45;vec2 gu=.5+(uv-.5)*k;float inb=step(0.,gu.x)*step(gu.x,1.)*step(0.,gu.y)*step(gu.y,1.);
      gh+=inb*vec3(max(toLin(texture(uTex,gu*1.01-.005).rgb).r-.9,0.),max(toLin(texture(uTex,gu).rgb).g-.9,0.),max(toLin(texture(uTex,gu*.99+.005).rgb).b-.9,0.))/(1.+float(i));}
      c+=gh*flare*1.5*vec3(1.,.8,.6);}
    if(streak>0.){vec3 s2=vec3(0);for(int i=1;i<=12;i++){float o2=float(i)*uPx.x*10.;s2+=(max(toLin(texture(uTex,uv+vec2(o2,0)).rgb)-.7,0.)+max(toLin(texture(uTex,uv-vec2(o2,0)).rgb)-.7,0.))/float(i);}c+=s2*streak*.18*mix(vec3(.35,.6,1.),vec3(1.,.7,.3),streakT*.5+.5);}
  }
  c=grade(c);
  if(lutOn>.5)c=mix(c,tet(uLut,lutN,c),lutMix);
  if(lutOn2>.5)c=mix(c,tet(uLut2,lutN2,c),lutMix2);
  if(ptOn>.5)c=mix(c,vec3(texture(uPT,vec2((c.r*255.+.5)/256.,.5)).r,texture(uPT,vec2((c.g*255.+.5)/256.,.5)).g,texture(uPT,vec2((c.b*255.+.5)/256.,.5)).b),ptAmt);
  if(lab>.5)c=toSrgb(labFilm(toLin(c)));
  if(lab>.5&&bake<.5){float cell=floor(uv.x*420.),hh=h(vec2(cell,floor(t*.4)));
    c+=step(1.-scr*.02,hh)*step(.35,h(vec2(cell,9.)))*(1.-smoothstep(0.,.14,abs(fract(uv.x*420.)-.5)))*.4;
    c*=1.+strk*.12*sin(uv.x*90.+sin(uv.y*3.)*2.)*sin(uv.x*13.);
    c*=1.-step(1.-dust*.004,h(floor(uv*vec2(320.,180.))+floor(t*.3)))*.7;}
  if(neg>.5&&bake<.5)c=pow(1.-c,vec3(.8))*mix(vec3(1.),vec3(1.,.6,.32),mask);
  if(bake<.5){
    vec2 q=uv-.5;c*=1.-vig*smoothstep(.25,.85,length(q)*1.2);
    if(grain>0.){vec2 p=uv/uPx*mix(.6,.22,gpush);vec2 i0=floor(p),f0=fract(p);float dm=2.,id=0.;
      for(int j=-1;j<=1;j++)for(int k=-1;k<=1;k++){vec2 g0=vec2(k,j),cc=i0+g0,o=vec2(h(cc+t),h(cc+t+7.));float d=length(g0+o-f0);if(d<dm){dm=d;id=h(cc+t+3.);}}
      float L2=luma(c);c+=(id-.5)*smoothstep(1.,.15,dm)*grain*.45*4.*L2*(1.-L2);}
  }
  if(bake<.5){float L=luma(c);
    if(fc>.5)c=L<.03?vec3(.5,0,.7):L>.97?vec3(1,0,0):L>.9?vec3(1,.9,0):(L>.4&&L<.5)?vec3(0,.7,.2):(L>.6&&L<.7)?vec3(.9,.5,.6):vec3(L);
    if(zhud>.5){float zq=5.+log2(max(pow(L,2.2),1e-4)/.18);float ln=1.-smoothstep(0.,.05,min(fract(zq),1.-fract(zq)));c=mix(c,vec3(1.),ln*.55);if(floor(zq)==zsel)c=mix(c,vec3(1.,.3,.5),.28);}
    if(zeb>.5&&L>.95&&mod(gl_FragCoord.x+gl_FragCoord.y,12.)<5.)c=vec3(1,0,0);
    if(frame>0.){vec2 d=uv-.5;if(frame>ia){if(abs(d.y)>.5*ia/frame)c=vec3(0);}else if(abs(d.x)>.5*frame/ia)c=vec3(0);}}
  if(zone>=0.&&bake<.5&&abs(luma(c)-zone)<.035)c=mix(c,vec3(1.,.15,.35),.65);
  o=vec4(clamp(c,0.,1.),1);}`;
function prog(){const mk=(t,s)=>{const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))throw gl.getShaderInfoLog(x);return x};
  const P=gl.createProgram();gl.attachShader(P,mk(gl.VERTEX_SHADER,VS));gl.attachShader(P,mk(gl.FRAGMENT_SHADER,FS));gl.linkProgram(P);return P}
const P=prog();gl.useProgram(P);
const vb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,vb);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,1,1]),gl.STATIC_DRAW);
const loc=gl.getAttribLocation(P,'p');gl.enableVertexAttribArray(loc);gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
const U=n=>gl.getUniformLocation(P,n);

function tex2D(unit){const t=gl.createTexture();gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);
  for(const [k,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_2D,k,v);return t}
const imgTex=tex2D(0), idTex=tex2D(2);
const lutTex=gl.createTexture();gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,lutTex);
for(const [k,v] of [[gl.TEXTURE_MIN_FILTER,gl.LINEAR],[gl.TEXTURE_MAG_FILTER,gl.LINEAR],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_R,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_3D,k,v);
gl.uniform1i(U('uTex'),0);gl.uniform1i(U('uLut'),1);

// ---------- state ----------
const DEF={ev:0,con:0,hi:0,sh:0,temp:0,tint:0,sat:0,vig:0,halo:0,grain:0,mono:0};
const RANGE={ev:[-4,4,.01,'Exposure'],con:[-1,1,.01,'Contrast'],hi:[-1,1,.01,'Highlights'],sh:[-1,1,.01,'Shadows'],temp:[-1,1,.01,'Temp'],tint:[-1,1,.01,'Tint'],sat:[-1,1,.01,'Saturation'],vig:[0,1,.01,'Vignette'],halo:[0,1,.01,'Halation'],grain:[0,1,.01,'Grain'],mono:[0,1,.01,'Mono']};
Object.assign(DEF,{clar:0,dens:0,gnd:0,gndY:0,barrel:0,ca:0,streak:0,cpl:0,mask:0,subj:0,bg:0,skin:0,lx:0,ly:0,lint:0,lwarm:0,ptR:0,ptG:0,ptB:0,dev:0,silver:0,scr:0,strk:0,dust:0,gwv:0,stretch:0,starmin:0,grad:0,dyeC:0,dyeM:0,dyeY:0,smear:0,sang:0,flare:0,streakT:0,gpush:0,mel:0,hemo:0,dering:0,adjc:0,hsW:0,hsShift:0,hsSat:0,ptAmt:0,mEv:0,mCon:0,mSat:0,mTmp:0,mF:0,gAmt:0,gScale:0,gAng:0,gSoft:0,gWarm:0});
Object.assign(RANGE,{clar:[-1,1,.01,'Clarity'],dens:[0,1,.01,'Density'],gnd:[0,4,.01,'GND stops'],gndY:[-.5,.5,.01,'GND horizon'],barrel:[-1,1,.01,'Barrel'],ca:[0,2,.01,'Fringing'],streak:[0,1,.01,'Streak'],cpl:[0,1,.01,'Polarizer'],mask:[0,1,.01,'Orange mask'],subj:[-3,3,.01,'Subject EV'],bg:[-3,3,.01,'Backdrop EV'],skin:[-1,1,.01,'Skin warmth'],lx:[-.5,.5,.01,'Light X'],ly:[-.5,.5,.01,'Light Y'],lint:[0,3,.01,'Intensity'],lwarm:[-1,1,.01,'Light warmth'],ptR:[-12,12,.25,'Printer R'],ptG:[-12,12,.25,'Printer G'],ptB:[-12,12,.25,'Printer B'],dev:[-1,1,.01,'Push / pull'],silver:[0,1,.01,'Silver retain'],scr:[0,1,.01,'Scratches'],strk:[0,1,.01,'Dev streaks'],dust:[0,1,.01,'Dust'],gwv:[0,1,.01,'Gate weave'],stretch:[0,1,.01,'Stretch'],starmin:[0,1,.01,'Star shrink'],grad:[0,1,.01,'Gradient removal'],dyeC:[-1,1,.01,'Cyan dye'],dyeM:[-1,1,.01,'Magenta dye'],dyeY:[-1,1,.01,'Yellow dye'],smear:[0,1,.01,'Shutter angle'],sang:[-1,1,.01,'Smear direction'],flare:[0,1,.01,'Iris ghosts'],streakT:[-1,1,.01,'Streak tint'],gpush:[0,1,.01,'Push grain'],mel:[-1,1,.01,'Melanin'],hemo:[-1,1,.01,'Hemoglobin'],dering:[0,1,.01,'De-ring'],adjc:[0,1,.01,'Adjacency'],hsW:[0,.3,.005,'Slice width'],hsShift:[-1,1,.01,'Hue twist'],hsSat:[-1,1,.01,'Slice saturation'],ptAmt:[0,1,.01,'Palette amount'],mEv:[-3,3,.01,'Mask exposure'],mCon:[-1,1,.01,'Mask contrast'],mSat:[-1,1,.01,'Mask saturation'],mTmp:[-1,1,.01,'Mask warmth'],mF:[0,1,.01,'Mask feather'],gAmt:[0,1,.01,'Gobo strength'],gScale:[0,1,.01,'Pattern scale'],gAng:[-1,1,.01,'Pattern angle'],gSoft:[0,1,.01,'Softness'],gWarm:[-1,1,.01,'Light warmth']});
let S={...DEF,lutOn:0,lutMix:1}, W=0,H=0, hasImg=false;
const PRESETS={
 'Vision3 500T':{ev:.1,con:.12,sh:.25,temp:-.25,tint:.05,sat:-.05,halo:.35,grain:.4},
 'Eterna 250D':{con:-.05,sh:.2,sat:-.25,temp:.05,tint:-.05,grain:.25},
 'Bleach Bypass':{con:.45,sat:-.55,hi:-.15,grain:.45},
 'Nordic Cold':{temp:-.5,tint:-.1,sat:-.3,con:.1,sh:.1},
 'Neo-Western Amber':{temp:.55,con:.2,sat:.1,vig:.35,halo:.2,grain:.3},
 'Tokyo Neon':{temp:-.2,tint:.4,sat:.35,con:.2,halo:.55,vig:.25},
 'Tri-X 400':{mono:1,con:.35,grain:.85,vig:.2,hi:-.1},
 'Ilford HP5':{mono:1,con:.2,grain:.6,sh:.2},
 'Kodachrome 64':{temp:.3,sat:.35,con:.25,vig:.2,grain:.2},'Technicolor 3-Strip':{sat:.6,con:.2,temp:.1,dens:.4,halo:.15},
 'Anamorphic Blue':{streak:.7,halo:.3,temp:-.15,con:.15,vig:.2},'Cooke Speed Panchro':{barrel:.2,ca:.5,vig:.4,halo:.25,sat:-.1,grain:.3,clar:-.2}};

// ---------- UI ----------
const sl=$('#sliders'),outs={},ins={};
const GR={Light:'ev con hi sh gnd gndY',Color:'temp tint sat dens cpl',Effects:'vig halo grain mono streak',Lens:'clar barrel ca',Lab:'ptR ptG ptB mask dev silver scr strk dust gwv',Masks:'subj bg skin mEv mCon mSat mTmp mF',Gobo:'gAmt gScale gAng gSoft gWarm',Relight:'lx ly lint lwarm',Astro:'stretch starmin grad','Dye density':'dyeC dyeM dyeY',Shutter:'smear sang',Flare:'flare streakT',Grain:'gpush',Skin:'mel hemo',Edge:'dering adjc',Hue:'hsW hsShift hsSat',Palette:'ptAmt'},boxes={};
for(const g in GR){const d=document.createElement('details');d.open=true;d.dataset.t=({Light:'light',Color:'color',Effects:'film',Lens:'lens',Lab:'lab',Cinema:'cinema',Masks:'masks',Relight:'relight',Astro:'astro',Hue:'hue',Palette:'palette',Gobo:'gobo'})[g]||'pro';if(d.dataset.t=='pro')d.innerHTML='<summary>'+g+'</summary>';sl.append(d);GR[g].split(' ').forEach(k=>boxes[k]=d)}
const grp=k=>boxes[k]||sl;
for(const k in RANGE){const [a,b,s,n]=RANGE[k];const r=document.createElement('label');r.className='row';
  r.innerHTML=`<span>${n}</span><input type=range min=${a} max=${b} step=${s} value=0 data-k=${k}><output>0</output>`;grp(k).append(r);
  ins[k]=r.querySelector('input');outs[k]=r.querySelector('output');
  ins[k].addEventListener('input',e=>{S[k]=+e.target.value;sync(k);draw()});
  ins[k].addEventListener('pointerdown',()=>document.body.classList.add('ghost')); // Ghost Workspace
}
addEventListener('pointerup',()=>document.body.classList.remove('ghost'));
const sync=k=>outs[k].textContent=(+S[k]).toFixed(2);
function setAll(o){for(const k in DEF){S[k]=o[k]??0;ins[k].value=S[k];sync(k)}draw()}
for(const n in PRESETS){const b=document.createElement('button');b.className='btn';b.textContent=n;b.onclick=()=>setAll(PRESETS[n]);$('#presets').append(b)}
$('#reset').onclick=()=>{S.lutOn=0;setAll({})};
$('#lutMix').oninput=e=>{S.lutMix=+e.target.value;draw()};

// ---------- image ----------
$('#file').onchange=async e=>{const f=e.target.files[0];if(!f)return;
  const im=await createImageBitmap(f,{imageOrientation:'from-image'});srcIm=im;srcURL=URL.createObjectURL(f);srcC=null;MASKS.forEach(m=>m.t=0);mAct=-1;BR=null;idb('put','src',f);tixAdd('open');try{localStorage.setItem('photos',lsn('photos')+1)}catch(e){}
  const sc=Math.min(1,(matchMedia('(pointer:coarse)').matches?2048:4096)/Math.max(im.width,im.height));W=Math.round(im.width*sc);H=Math.round(im.height*sc);
  cv.width=W;cv.height=H;gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,imgTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);{const c2=document.createElement('canvas');c2.width=W;c2.height=H;c2.getContext('2d').drawImage(im,0,0,W,H);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,c2)}gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
  hasImg=true;document.body.classList.add('has');setTool(lastToolSaved());cv.classList.remove('reveal');void cv.offsetWidth;cv.classList.add('reveal');dev();HI.length=0;hi=-1;snap();setTimeout(()=>{buildThumbs();moodLook()},60);setTimeout(tour,900);$('#empty').remove();fit();draw()};

// ---------- .cube parsing ----------
$('#lut').onchange=async e=>{const f=e.target.files[0];if(!f)return;
  let N=0,d=[];for(const ln of (await f.text()).split(/\r?\n/)){const t=ln.trim();if(!t||t[0]=='#')continue;
    if(t.startsWith('LUT_3D_SIZE'))N=+t.split(/\s+/)[1];else if(/^[-\d.]/.test(t)){const v=t.split(/\s+/).map(Number);if(v.length>=3)d.push(v[0],v[1],v[2],1)}}
  if(!N||d.length!==N**3*4){alert('Unsupported or malformed .cube (needs LUT_3D_SIZE).');return}
  gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_3D,lutTex);
  gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA16F,N,N,N,0,gl.RGBA,gl.FLOAT,new Float32Array(d));
  S.lutN=N;S.lutOn=1;draw()};
// NOTE: trilinear sampling here; swap in tetrahedral interpolation in the shader for production.

// ---------- render ----------
let bakeMode=0;
function setU(){const sets={...Object.fromEntries(Object.keys(DEF).map(k=>[k,S[k]])),aud:AUD,orig:S.orig||0,relOn:S.relOn||0,gType:S.gType||0,lab:S.lab||0,hsOn:S.hsOn||0,hsHue:S.hsHue||0,ptOn:S.ptOn||0,zhud:S.zhud||0,zsel:S.zsel??-1,negS:S.negS||0,prtS:S.prtS||0,labView:S.labView||0,zone:S.zone??-1,maskOn:S.maskOn||0,neg:S.neg||0,lutOn2:S.lutOn2||0,lutMix2:S.lutMix2??1,zeb:S.zeb||0,fc:S.fc||0,frame:S.frame||0,ia:W*sq/H,ev:S.ev,con:S.con,hi:S.hi,sh:S.sh,temp:S.temp,tint:S.tint,sat:S.sat,vig:S.vig,halo:S.halo,grain:S.grain,mono:S.mono,lutOn:S.lutOn,lutMix:S.lutMix,bake:bakeMode,t:Math.random()*100};
  for(const k in sets)gl.uniform1f(U(k),sets[k]);gl.uniform2f(U('uPx'),1/W,1/H);gl.uniform1i(U('uCrv'),3);gl.uniform1i(U('uMask'),5);gl.uniform1i(U('uDepth'),6);gl.uniform3fv(U('bgc[0]'),BGC);packMasks();gl.uniform1i(U('uBrush'),8);gl.uniform4fv(U('mA[0]'),MA);gl.uniform4fv(U('mB[0]'),MB);gl.uniform4fv(U('mC[0]'),MC);gl.uniform1f(U('mShow'),S.mShow??-1);gl.uniform1i(U('uPT'),7);gl.uniform1fv(U('zadj[0]'),ZA);gl.uniform3fv(U('hsl[0]'),HSLA);gl.uniform1f(U('lutN'),S.lutN||2);gl.uniform1i(U('uLut2'),4);gl.uniform1f(U('lutN2'),S.lutN2||2);for(const k in WH)gl.uniform3fv(U(k),WH[k])}
function draw(){if(!hasImg)return;gl.viewport(0,0,W,H);gl.uniform1i(U('uTex'),0);setU();gl.drawArrays(gl.TRIANGLE_STRIP,0,4);if(!S.anim&&document.body.classList.contains('sc'))scope();rbUpdate();guideUpdate()}
let sm=0;const SM=['Waveform','RGB Parade','Vectorscope'];
function scope(){const c=$('#scope'),x=c.getContext('2d'),w=c.width,h=c.height,R=h/2-4;c.title=SM[sm]+' (click to switch)';x.fillStyle='#000';x.fillRect(0,0,w,h);
 x.strokeStyle='rgba(255,255,255,.2)';x.beginPath();
 if(sm<2)for(const v of[0,.25,.5,.75,1]){x.moveTo(0,h-1-v*(h-1));x.lineTo(w,h-1-v*(h-1))}
 else{for(const k of[.75,1]){x.moveTo(w/2+R*k,h/2);x.arc(w/2,h/2,R*k,0,7)}x.moveTo(w/2,h/2);x.lineTo(w/2+Math.cos(2.147)*R,h/2-Math.sin(2.147)*R)}
 x.stroke();
 const rh=72,row=new Uint8Array(W*4),st=Math.max(1,Math.floor(W/256));
 for(let j=0;j<rh;j++){gl.readPixels(0,Math.floor(j*(H-1)/rh),W,1,gl.RGBA,gl.UNSIGNED_BYTE,row);
  for(let i=0;i<W;i+=st){const k=i*4,r=row[k]/255,g=row[k+1]/255,bl=row[k+2]/255,y=.2126*r+.7152*g+.0722*bl,X=i/W*w;
   if(sm==0){x.fillStyle='rgba(120,255,140,.2)';x.fillRect(X,h-1-y*(h-1),1,1)}
   else if(sm==1){[[r,'rgba(255,70,70,.25)'],[g,'rgba(70,255,100,.25)'],[bl,'rgba(80,140,255,.25)']].forEach(([v,col],n)=>{x.fillStyle=col;x.fillRect(X/3+n*w/3,h-1-v*(h-1),1,1)})}
   else{x.fillStyle='rgba(255,200,120,.35)';x.fillRect(w/2+(bl-y)/1.8556*2*R,h/2-(r-y)/1.5748*2*R,1,1)}}}}
$('#scope').onclick=e=>{if(sm==0){const r=e.target.getBoundingClientRect(),z=1-(e.clientY-r.top)/r.height;S.zone=(S.zone>=0&&Math.abs(S.zone-z)<.05)?-1:z;draw()}else{sm=(sm+1)%3;scope()}};$('#scope').ondblclick=()=>{sm=(sm+1)%3;scope()};
// ---------- exports ----------
const dl=(b,n)=>{wrapUp(n);if(!($('#fun')&&$('#fun').classList.contains('on'))&&/\.(png|jpg|webm)$/.test(n)){tixAdd('export');rollAdvance()}if(window.LuminaNative){const r=new FileReader();r.onload=()=>LuminaNative.save(r.result.split(',')[1],n,b.type||'application/octet-stream');r.readAsDataURL(b);return}const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=n;a.click()};
$('#exPng').onclick=()=>{if(hasImg){draw();cv.toBlob(b=>dl(b,fname('.png')))}};
$('#exCube').onclick=()=>{ // bake grade (+LUT) into a 33^3 cube by running an identity lattice through the same shader
  const N=33,TW=256,TH=Math.ceil(N**3/TW),id=new Uint8Array(TW*TH*4);
  for(let i=0;i<TW*TH;i++){const r=i%N,g=Math.floor(i/N)%N,b=Math.floor(i/(N*N));id.set([r,g,b].map(v=>Math.round(v/(N-1)*255)).concat(255),i*4)}
  const w0=cv.width,h0=cv.height;cv.width=TW;cv.height=TH;gl.viewport(0,0,TW,TH);
  gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,idTex);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,TW,TH,0,gl.RGBA,gl.UNSIGNED_BYTE,id);
  gl.uniform1i(U('uTex'),2);bakeMode=1;setU();gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  const out=new Uint8Array(TW*TH*4);gl.readPixels(0,0,TW,TH,gl.RGBA,gl.UNSIGNED_BYTE,out);
  let s=`TITLE "Lumina Bake"\nLUT_3D_SIZE ${N}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n`;
  for(let i=0;i<N**3;i++)s+=`${(out[i*4]/255).toFixed(6)} ${(out[i*4+1]/255).toFixed(6)} ${(out[i*4+2]/255).toFixed(6)}\n`;
  bakeMode=0;cv.width=w0;cv.height=h0;gl.uniform1i(U('uTex'),0);draw();dl(new Blob([s]),'lumina_bake.cube')};

// ---------- curves (master + R/G/B, up to 8 points each) ----------
const crvTex=tex2D(3),CH=['m','r','g','b'],PT={};CH.forEach(k=>PT[k]=[[0,0],[1,1]]);let cur='m',drag=-1;
function curveLUT(p){p=[...p].sort((a,b)=>a[0]-b[0]);const n=p.length,d=[],m=[];
 for(let i=0;i<n-1;i++)d[i]=(p[i+1][1]-p[i][1])/Math.max(1e-4,p[i+1][0]-p[i][0]);
 m[0]=d[0];m[n-1]=d[n-2];for(let i=1;i<n-1;i++)m[i]=d[i-1]*d[i]<=0?0:(d[i-1]+d[i])/2;
 const o=new Float32Array(256);
 for(let k=0;k<256;k++){const x=k/255;let i=0;while(i<n-2&&x>p[i+1][0])i++;const h=Math.max(1e-4,p[i+1][0]-p[i][0]),t=Math.min(1,Math.max(0,(x-p[i][0])/h)),t2=t*t,t3=t2*t;
  o[k]=Math.min(1,Math.max(0,(2*t3-3*t2+1)*p[i][1]+(t3-2*t2+t)*h*m[i]+(-2*t3+3*t2)*p[i+1][1]+(t3-t2)*h*m[i+1]))}return o}
function updateCurves(){const M=curveLUT(PT.m),R=curveLUT(PT.r),G=curveLUT(PT.g),B=curveLUT(PT.b),d=new Uint8Array(1024);
 const f=(a,v)=>Math.round(a[Math.round(v*255)]*255);
 for(let k=0;k<256;k++)d.set([f(M,R[k]),f(M,G[k]),f(M,B[k]),255],k*4);
 gl.activeTexture(gl.TEXTURE3);gl.bindTexture(gl.TEXTURE_2D,crvTex);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,256,1,0,gl.RGBA,gl.UNSIGNED_BYTE,d);draw()}
const cc=$('#crv'),cx=cc.getContext('2d');
function drawCrv(){cx.clearRect(0,0,200,200);cx.strokeStyle='rgba(255,255,255,.15)';cx.beginPath();for(let i=1;i<4;i++){cx.moveTo(i*50,0);cx.lineTo(i*50,200);cx.moveTo(0,i*50);cx.lineTo(200,i*50)}cx.stroke();
 const col={m:'#fff',r:'#ff5a5a',g:'#5aff7a',b:'#5a9bff'}[cur],L=curveLUT(PT[cur]);cx.strokeStyle=col;cx.fillStyle=col;cx.beginPath();
 for(let k=0;k<256;k++){const X=k/255*200,Y=200-L[k]*200;k?cx.lineTo(X,Y):cx.moveTo(X,Y)}cx.stroke();
 for(const p of PT[cur]){cx.beginPath();cx.arc(p[0]*200,200-p[1]*200,5,0,7);cx.fill()}}
const cpos=e=>{const r=cc.getBoundingClientRect();return[Math.min(1,Math.max(0,(e.clientX-r.left)/r.width)),Math.min(1,Math.max(0,1-(e.clientY-r.top)/r.height))]};
cc.onpointerdown=e=>{const q=cpos(e),P=PT[cur];drag=P.findIndex(p=>Math.hypot(p[0]-q[0],p[1]-q[1])<.06);
 if(drag<0&&P.length<8){P.push(q);drag=P.length-1}cc.setPointerCapture(e.pointerId);drawCrv()};
cc.onpointermove=e=>{if(drag<0)return;PT[cur][drag]=cpos(e);drawCrv();updateCurves()};
cc.onpointerup=()=>drag=-1;
cc.ondblclick=e=>{const q=cpos(e),P=PT[cur],i=P.findIndex(p=>Math.hypot(p[0]-q[0],p[1]-q[1])<.06);if(i>=0&&P.length>2){P.splice(i,1);drawCrv();updateCurves()}};
document.querySelectorAll('#crvTabs button').forEach(b=>b.onclick=()=>{cur=b.dataset.c;document.querySelectorAll('#crvTabs button').forEach(x=>x.classList.toggle('on',x===b));drawCrv()});
// ---------- 8-band HSL mixer ----------
const HSLA=new Float32Array(24),hb=$('#hslBand'),hs=[...document.querySelectorAll('.hs')];
['Red','Orange','Yellow','Green','Aqua','Blue','Purple','Magenta'].forEach((n,i)=>hb.add(new Option(n,i)));
hs.forEach(s=>s.oninput=()=>{HSLA[hb.value*3+ +s.dataset.i]=+s.value;draw()});
hb.onchange=()=>hs.forEach(s=>s.value=HSLA[hb.value*3+ +s.dataset.i]);
$('#reset').addEventListener('click',()=>{CH.forEach(k=>PT[k]=[[0,0],[1,1]]);HSLA.fill(0);hs.forEach(s=>s.value=0);drawCrv();updateCurves()});
updateCurves();drawCrv();

// ---------- slot 2 LUT, wheels, overlays, framing, de-squeeze, edit files ----------
const lutTex2=gl.createTexture();gl.activeTexture(gl.TEXTURE4);gl.bindTexture(gl.TEXTURE_3D,lutTex2);
for(const k of[gl.TEXTURE_WRAP_S,gl.TEXTURE_WRAP_T,gl.TEXTURE_WRAP_R])gl.texParameteri(gl.TEXTURE_3D,k,gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_3D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
$('#lut2').onchange=async e=>{const f=e.target.files[0];if(!f)return;let N=0,d=[];
  for(const ln of(await f.text()).split(/\r?\n/)){const t=ln.trim();if(!t||t[0]=='#')continue;if(t.startsWith('LUT_3D_SIZE'))N=+t.split(/\s+/)[1];else if(/^[-\d.]/.test(t)){const v=t.split(/\s+/).map(Number);if(v.length>=3)d.push(v[0],v[1],v[2],1)}}
  if(!N||d.length!==N**3*4){alert('Unsupported .cube');return}
  gl.activeTexture(gl.TEXTURE4);gl.bindTexture(gl.TEXTURE_3D,lutTex2);gl.texImage3D(gl.TEXTURE_3D,0,gl.RGBA16F,N,N,N,0,gl.RGBA,gl.FLOAT,new Float32Array(d));S.lutN2=N;S.lutOn2=1;draw()};
$('#lutMix2').oninput=e=>{S.lutMix2=+e.target.value;draw()};
const WH={lift:[0,0,0],gam:[0,0,0],gain:[0,0,0]};
document.querySelectorAll('.wh').forEach(i=>i.oninput=()=>{const v=parseInt(i.value.slice(1),16);WH[i.dataset.w]=[v>>16&255,v>>8&255,v&255].map(x=>(x-128)/127);draw()});
$('#zeb').onchange=e=>{S.zeb=+e.target.checked;draw()};$('#fc').onchange=e=>{S.fc=+e.target.checked;draw()};
$('#frame').onchange=e=>{S.frame=+e.target.value;draw()};
let sq=1;$('#sq').onchange=e=>{sq=+e.target.value;fit();draw()};
function fit(){const vw=innerWidth,vh=innerHeight*(innerWidth<700&&innerHeight>innerWidth?.6:1)-(innerHeight<500?60:0),a=W*sq/H,w=Math.min(vw,vh*a),h=w/a;Object.assign(cv.style,{objectFit:'fill',width:w+'px',height:h+'px',left:(vw-w)/2+'px',top:(vh-h)/2+'px'})}
addEventListener('resize',()=>{if(hasImg){fit();guideUpdate();rbUpdate()}});
$('#reset').addEventListener('click',()=>{for(const k in WH)WH[k]=[0,0,0];document.querySelectorAll('.wh').forEach(i=>i.value='#808080');S.lutOn2=0;draw()});
$('#saveJ').onclick=()=>dl(new Blob([JSON.stringify({S,PT,HSLA:[...HSLA],WH,MK:MASKS})]),'lumina_edit.json');
$('#loadJ').onchange=async e=>{const j=JSON.parse(await e.target.files[0].text());Object.assign(S,j.S,{lutOn:0,lutOn2:0});for(const k of CH)PT[k]=j.PT[k];HSLA.set(j.HSLA);Object.assign(WH,j.WH);
  for(const k in DEF){ins[k].value=S[k];sync(k)}updateCurves();drawCrv();hb.onchange()};

// ---------- audio-reactive timing, reference match, rebate border, sample ----------
let AUD=0,an=null;
$('#audio').onchange=async e=>{if(!e.target.checked){an=null;AUD=0;draw();return}
 try{const st=await navigator.mediaDevices.getUserMedia({audio:true}),ac=new AudioContext();an=ac.createAnalyser();an.fftSize=256;ac.createMediaStreamSource(st).connect(an);
  const b=new Uint8Array(128);let sm=0;(function tick(){if(!an)return;an.getByteFrequencyData(b);let s=0;for(const v of b)s+=v;sm+=(s/b.length/255-sm)*.08;AUD=(sm-.15)*3;draw();requestAnimationFrame(tick)})()}
 catch{e.target.checked=false}};
function meanRGB(src){const c=document.createElement('canvas');c.width=c.height=32;const x=c.getContext('2d');x.drawImage(src,0,0,32,32);const d=x.getImageData(0,0,32,32).data;let r=0,g=0,b=0;for(let i=0;i<d.length;i+=4){r+=d[i];g+=d[i+1];b+=d[i+2]}const n=1024*255;return[r/n,g/n,b/n]}
$('#ref').onchange=async e=>{if(!hasImg)return;const t=meanRGB(await createImageBitmap(e.target.files[0])),c=meanRGB(cv),L=v=>.2126*v[0]+.7152*v[1]+.0722*v[2];
 S.ev+=Math.log2((L(t)+.02)/(L(c)+.02))*.8;S.temp+=((t[0]-t[2])-(c[0]-c[2]))*1.5;S.tint-=((t[1]-(t[0]+t[2])/2)-(c[1]-(c[0]+c[2])/2))*1.5;
 for(const k of['ev','temp','tint']){S[k]=Math.max(RANGE[k][0],Math.min(RANGE[k][1],S[k]));ins[k].value=S[k];sync(k)}draw()};
$('#exPng').onclick=()=>{if(!hasImg)return;draw();if(!$('#reb').checked)return cv.toBlob(b=>dl(b,'lumina.png'));
 const p=Math.round(W*.05),o=document.createElement('canvas');o.width=W+2*p;o.height=H+4*p;const x=o.getContext('2d');x.fillStyle='#1b120a';x.fillRect(0,0,o.width,o.height);x.drawImage(cv,p,2*p);
 x.fillStyle='#e9d9b0';for(let i=0;i<o.width/(p*.9);i++)for(const y of[p*.45,o.height-p*1.2])x.fillRect(i*p*.9+p*.2,y,p*.5,p*.7);
 x.font=`${p*.45}px monospace`;x.fillText('KODAK 5219   24   24A',p,2*p-p*.2);o.toBlob(b=>dl(b,'lumina_rebate.png'))};
function sampleCanvas(){const c=document.createElement('canvas');c.width=1600;c.height=900;const x=c.getContext('2d');
 let g=x.createLinearGradient(0,0,0,640);g.addColorStop(0,'#1c2a52');g.addColorStop(.55,'#c8627a');g.addColorStop(1,'#ffb36b');x.fillStyle=g;x.fillRect(0,0,1600,900);
 x.fillStyle='#fff';for(let i=0;i<60;i++)x.fillRect(Math.random()*1600,Math.random()*260,1.5,1.5);
 g=x.createRadialGradient(1050,540,0,1050,540,280);g.addColorStop(0,'#fff');g.addColorStop(.12,'#fff3c4');g.addColorStop(1,'rgba(255,180,100,0)');x.fillStyle=g;x.fillRect(0,0,1600,900);
 x.fillStyle='#10131c';x.beginPath();x.moveTo(0,900);for(let i=0;i<=16;i++)x.lineTo(i*100,560+Math.sin(i*1.7)*70+Math.cos(i*.6)*45);x.lineTo(1600,900);x.fill();
 g=x.createLinearGradient(0,660,0,900);g.addColorStop(0,'#e58a6e');g.addColorStop(1,'#1d2a4a');x.fillStyle=g;x.fillRect(0,660,1600,240);
 g=x.createLinearGradient(0,660,0,900);g.addColorStop(0,'rgba(255,245,200,.9)');g.addColorStop(1,'rgba(255,200,120,0)');x.fillStyle=g;x.fillRect(1010,660,80,240);
 return c}
$('#sample').onclick=async()=>{await $('#file').onchange({target:{files:[await new Promise(r=>sampleCanvas().toBlob(r))]}})};

// ---------- dock + one-tool-at-a-time tray ----------
function setTool(t){document.body.dataset.tool=t;$('#panel').hidden=!t;
 document.querySelectorAll('#panel [data-t]').forEach(e=>e.hidden=e.dataset.t!==t);
 document.querySelectorAll('#dock button').forEach(b=>{const on=b.dataset.t===t;b.classList.toggle('on',on);if(on)$('#trayTitle').innerHTML=b.dataset.n})}
document.querySelectorAll('#dock button').forEach(b=>b.onclick=()=>setTool(document.body.dataset.tool===b.dataset.t?'':b.dataset.t));
document.querySelectorAll('#open,#open2').forEach(b=>b.onclick=()=>$('#file').click());
$('#scopeBtn').onclick=()=>{document.body.classList.toggle('sc');if(hasImg)scope()};

setTool('');

// ---------- FOR FILMMAKERS: six decks (parametric approximations, not licensed stock data) ----------
const DECKS={
'35mm Motion':{'Vision3 50D':{con:.1,sat:.05,temp:.05,grain:.15,halo:.1},'Vision3 250D':{con:.12,sat:-.05,temp:.08,grain:.25,halo:.15},'Vision3 500T':{ev:.1,con:.12,sh:.25,temp:-.25,sat:-.05,halo:.35,grain:.4},'Double-X 5222':{mono:1,con:.3,grain:.7,halo:.1},'Eterna 250D':{con:-.05,sh:.2,sat:-.25,temp:.05,grain:.25},'Eterna 500T':{temp:-.2,sat:-.2,sh:.2,grain:.4}},
'Vintage':{'Technicolor 3-Strip':{sat:.6,con:.2,temp:.1,dens:.4,halo:.15},'Kodachrome 64':{temp:.3,sat:.35,con:.25,vig:.2,grain:.2},'Agfacolor Neu':{temp:.15,tint:.2,sat:-.15,con:.15,grain:.3,vig:.2},'Bleach Bypass':{con:.45,sat:-.55,hi:-.15,grain:.45},'Autochrome':{sat:-.35,tint:.25,temp:.1,grain:.9,con:-.1,vig:.3,clar:-.3}},
'Arthouse':{'French New Wave':{sat:-.3,con:-.1,temp:-.05,ev:.2,grain:.4,halo:.2},'Nordic Cold':{temp:-.5,tint:-.1,sat:-.3,con:.1,sh:.1},'Tokyo Neon':{temp:-.2,tint:.4,sat:.35,con:.2,halo:.55,vig:.25},'Iranian Golden Hour':{temp:.5,sat:.15,hi:-.2,halo:.3,vig:.2},'A24 Daylight':{temp:.2,sat:-.1,con:.05,grain:.25,halo:.1}},
'Camera log (approx.)':{'ARRI LogC':{con:.55,sat:.45,sh:-.1,ev:.1},'Sony S-Log3':{con:.6,sat:.5,ev:.2},'RED IPP2':{con:.45,sat:.35},'Canon C-Log2':{con:.5,sat:.4,ev:.15},'BMD Film Gen5':{con:.5,sat:.4}},
'Genre':{'Neo-Western Amber':{temp:.55,con:.2,sat:.1,vig:.35,halo:.2,grain:.3},'Sci-Fi Sodium':{temp:.45,tint:.15,sat:-.1,con:.3,vig:.3,halo:.4},'High-Key Commercial':{ev:.4,con:-.1,sat:.1,sh:.4,hi:-.2},'Noir Colour':{con:.5,sat:-.4,vig:.45,sh:-.1,grain:.3},'Thriller Emerald':{tint:-.45,temp:-.15,sat:-.15,con:.3,vig:.4,grain:.3}},
'Black & White':{'Ilford HP5':{mono:1,con:.2,grain:.6,sh:.2},'Tri-X 400 +1':{mono:1,con:.5,grain:1,ev:.2,vig:.25},'Ortho Cinema':{mono:1,con:.3,hi:-.2,grain:.5,halo:.2},'Delta 100':{mono:1,con:.15,grain:.2}}};
(function(){const cl=(k,v)=>Math.max(RANGE[k][0],Math.min(RANGE[k][1],v)),V={
 push:p=>({...p,con:cl('con',(p.con||0)*1.25+.05),grain:cl('grain',(p.grain||0)*1.4+.1),ev:cl('ev',(p.ev||0)+.1)}),
 pull:p=>({...p,con:cl('con',(p.con||0)*.8),grain:cl('grain',(p.grain||0)*.6),ev:cl('ev',(p.ev||0)-.05),sat:cl('sat',(p.sat||0)*.9)}),
 warm:p=>({...p,temp:cl('temp',(p.temp||0)+.2)}),cool:p=>({...p,temp:cl('temp',(p.temp||0)-.2)})};
 for(const d in DECKS){const o={};for(const n in DECKS[d]){o[n]=DECKS[d][n];for(const v in V)o[n+' \u00b7 '+v]=V[v](DECKS[d][n])}DECKS[d]=o}})();
let deck='35mm Motion',cp=null,srcIm=null,seg=null;const THU={},TH={};
function applyCP(){if(!cp)return;const o={};for(const k in cp)o[k]=Math.max(RANGE[k][0],Math.min(RANGE[k][1],cp[k]*$('#str').value));setAll(o)}
function renderDeck(){const D=$('#decks'),P=$('#deck');D.innerHTML=P.innerHTML='';
 for(const d in DECKS){const b=document.createElement('button');b.className='btn'+(d==deck?' on':'');b.textContent=d;b.onclick=()=>{deck=d;renderDeck();setTimeout(()=>hasImg&&buildThumbs(),30)};D.append(b)}
 for(const n in DECKS[deck]){const b=document.createElement('button');b.className='btn';b.textContent=n;b.dataset.n=n;b.dataset.k=(P.children.length+1)+'A';if(THU[n]){b.style.backgroundImage='linear-gradient(rgba(0,0,0,.3),rgba(0,0,0,.7)),url('+THU[n]+')';b.style.backgroundSize='cover'}P.append(b)}loopDeck()}
function loopDeck(){const P=$('#deck'),set=[...P.children];for(let i=0;i<2;i++)set.forEach(b=>P.append(b.cloneNode(true)));
 P.onscroll=()=>{const t=P.scrollWidth/3;if(t&&P.scrollLeft<t*.5)P.scrollLeft+=t;else if(t&&P.scrollLeft>t*1.5)P.scrollLeft-=t};
 P.onpointerenter=()=>{if(P.scrollLeft<5)P.scrollLeft=P.scrollWidth/3};
 P.onwheel=e=>{if(Math.abs(e.deltaY)>Math.abs(e.deltaX)){P.scrollLeft+=e.deltaY;e.preventDefault()}};
 P.onclick=e=>{const b=e.target.closest('button');if(b){cp=DECKS[deck][b.dataset.n];burn();clk();applyCP();slate('Look: '+b.dataset.n)}}}
$('#str').oninput=applyCP;$('#neg').onchange=e=>{S.neg=+e.target.checked;draw()};renderDeck();
// ---------- neural subject / skin mask (MediaPipe selfie multiclass, on-device) ----------
const maskTex=tex2D(5);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
$('#detect').onclick=async()=>{const st=$('#mstat');if(!srcIm){st.textContent='Open a photo first';return}
 try{st.textContent='Loading model…';
  if(!seg){const B='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14',V=await import(B+'/vision_bundle.mjs'),fs=await V.FilesetResolver.forVisionTasks(B+'/wasm');
   try{seg=await V.ImageSegmenter.createFromOptions(fs,SEGOPT('GPU'))}catch(e){seg=await V.ImageSegmenter.createFromOptions(fs,SEGOPT('CPU'))}}
  const m=seg.segment(small(srcIm,512)).categoryMask,w=m.width,h=m.height,a=m.getAsUint8Array(),d=new Uint8Array(w*h*4);
  for(let i=0;i<a.length;i++){d[i*4]=a[i]>0?255:0;d[i*4+1]=a[i]==2||a[i]==3?255:0;d[i*4+3]=255}lastMask={w,h,a:a.slice()};m.close();
  gl.activeTexture(gl.TEXTURE5);gl.bindTexture(gl.TEXTURE_2D,maskTex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,d);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
  S.maskOn=1;st.textContent='Subject found. Use the sliders above.';draw()}
 catch(e){st.textContent='Model could not load. It needs internet access and a normal page (it is blocked in the hosted preview).'}};

// ---------- ring cursor (mouse devices only) ----------
if(matchMedia('(hover:hover)').matches){const c=document.createElement('div');c.id='cur';document.body.append(c);let x=0,y=0,tx=0,ty=0;
 addEventListener('pointermove',e=>{tx=e.clientX;ty=e.clientY;c.style.opacity=1;c.classList.toggle('big',!!e.target.closest('button,select,label,summary,#crv'))});
 (function f(){x+=(tx-x)*.2;y+=(ty-y)*.2;c.style.transform='translate('+x+'px,'+y+'px)';requestAnimationFrame(f)})()}

// ================= v2: live thumbnails, gestures, history, sheet, tour, sphere, PWA =================
let AC=null;const clk=()=>{if(!$('#snd').checked)return;AC=AC||new AudioContext();const b=AC.createBuffer(1,2205,44100),d=b.getChannelData(0);for(let i=0;i<2205;i++)d[i]=(Math.random()*2-1)*Math.exp(-i/260);const s=AC.createBufferSource();s.buffer=b;s.connect(AC.destination);s.start()};
const burn=()=>{const b=$('#burn');b.style.setProperty('--x',(15+Math.random()*70)+'%');b.classList.remove('go');void b.offsetWidth;b.classList.add('go')};
let tt;const toast=t=>{const e=$('#toast');e.textContent=t;e.style.opacity=1;clearTimeout(tt);tt=setTimeout(()=>e.style.opacity=0,1100)};
// live look thumbnails: each preset rendered by the real shader at small size
function buildThumbs(){if(!hasImg)return;const tw=220,th=Math.round(tw*H/W),keep=S,z=Object.fromEntries(Object.keys(DEF).map(k=>[k,0]));
 for(const d of [deck])for(const n in DECKS[d]){S={...z,...DECKS[d][n],lutOn:0,lutOn2:0,lutMix:1,lutMix2:1,maskOn:0,neg:0,zone:-1,orig:0};gl.viewport(0,0,tw,th);setU();gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  const k=document.createElement('canvas');k.width=tw;k.height=th;k.getContext('2d').drawImage(cv,0,H-th,tw,th,0,0,tw,th);TH[n]=k;THU[n]=k.toDataURL('image/jpeg',.8)}
 S=keep;draw();renderDeck()}
$('#sheet').onclick=()=>{if(!hasImg)return;const names=Object.keys(DECKS[deck]),cols=3,tw=220,th=(TH[names[0]]||{height:150}).height,p=24,cw=tw+p,ch=th+p+30,rows=Math.ceil(names.length/cols),
 o=document.createElement('canvas');o.width=cols*cw+p;o.height=rows*ch+p*2+10;const x=o.getContext('2d');x.fillStyle='#0b0b0b';x.fillRect(0,0,o.width,o.height);x.fillStyle='#fff';x.font='bold 20px sans-serif';x.fillText('Lumina - '+deck,p,p+4);
 names.forEach((n,i)=>{const cx=p+(i%cols)*cw,cy=p*2+Math.floor(i/cols)*ch;x.fillStyle='#161616';x.fillRect(cx-6,cy-6,tw+12,th+36);if(TH[n])x.drawImage(TH[n],cx,cy);x.fillStyle='#aaa';x.font='13px monospace';x.fillText((i+1)+'A  '+n,cx,cy+th+18)});o.toBlob(b=>dl(b,'contact_sheet.png'))};
// gestures on the photo: hold = original, vertical drag = exposure, swipe = next look, pinch/wheel/double-tap = zoom
const PT2=new Map();let held=0,hold=0,g=null,zm=1,tx=0,ty=0;
const zoomApply=()=>{if(zm==1){tx=ty=0}cv.style.transform=zm==1?'':'translate('+tx+'px,'+ty+'px) scale('+zm+')'};
function stepLook(d){const ns=Object.keys(DECKS[deck]);let i=ns.findIndex(n=>DECKS[deck][n]===cp);i=(i+d+ns.length)%ns.length;cp=DECKS[deck][ns[i]];burn();clk();applyCP();slate('Look: '+ns[i])}
cv.addEventListener('pointerdown',e=>{cv.setPointerCapture(e.pointerId);PT2.set(e.pointerId,{x:e.clientX,y:e.clientY});g={x:e.clientX,y:e.clientY,t:Date.now(),m:'',ev:S.ev};
 if(PT2.size==1)hold=setTimeout(()=>{held=1;S.orig=1;draw()},350)});
cv.addEventListener('pointermove',e=>{const p=PT2.get(e.pointerId);if(!p)return;
 if(PT2.size==2){clearTimeout(hold);const o=[...PT2.values()].find(q=>q!==p),d0=Math.hypot(p.x-o.x,p.y-o.y),c0=[(p.x+o.x)/2,(p.y+o.y)/2];p.x=e.clientX;p.y=e.clientY;
  const d1=Math.hypot(p.x-o.x,p.y-o.y);if(d0>0)zm=Math.min(8,Math.max(1,zm*d1/d0));tx+=(p.x+o.x)/2-c0[0];ty+=(p.y+o.y)/2-c0[1];zoomApply();return}
 p.x=e.clientX;p.y=e.clientY;if(!g)return;const dx=p.x-g.x,dy=p.y-g.y;
 if(Math.hypot(dx,dy)>12){clearTimeout(hold);if(!g.m)g.m=Math.abs(dy)>Math.abs(dx)?'v':'h'}
 if(g.m=='v'&&zm==1){S.ev=Math.max(-4,Math.min(4,g.ev-dy/150));ins.ev.value=S.ev;sync('ev');draw()}});
const up=e=>{clearTimeout(hold);PT2.delete(e.pointerId);if(S.orig){S.orig=0;draw()}if(!g||PT2.size)return;const dx=e.clientX-g.x,dy=e.clientY-g.y;
 if(g.m=='h'&&Math.abs(dx)>70&&Date.now()-g.t<600&&zm==1)stepLook(dx<0?1:-1);else if(g.m=='v')snap();else if(pickArm&&Math.hypot(dx,dy)<8)pickAt(e);else if(!held&&Math.hypot(dx,dy)<8)setTool('');held=0;g=null};
cv.addEventListener('pointerup',up);cv.addEventListener('pointercancel',up);
cv.addEventListener('wheel',e=>{e.preventDefault();zm=Math.min(8,Math.max(1,zm*(e.deltaY<0?1.1:.9)));zoomApply()},{passive:false});
cv.addEventListener('dblclick',()=>{zm=zm>1?1:3;zoomApply()});
// haptics + double-tap a label to reset
const prev={};for(const k in DEF){prev[k]=0;ins[k].addEventListener('input',e=>{const v=+e.target.value;if(prev[k]*v<=0&&prev[k]!==v&&navigator.vibrate)navigator.vibrate(8);prev[k]=v});
 ins[k].previousElementSibling.ondblclick=()=>{S[k]=0;ins[k].value=0;sync(k);draw();snap()}}
// edit history scrubber
const HI=[],hist=$('#hist');let hi=-1;
const state=()=>JSON.stringify({S:{...S,orig:0},PT,HSLA:[...HSLA],WH,WS,MK:MASKS});
function applyState(j){Object.assign(S,j.S,{lutOn:S.lutOn,lutOn2:S.lutOn2});if(j.MK){MASKS=JSON.parse(JSON.stringify(j.MK));mAct=MASKS.findIndex(m=>m.t);syncMask()}for(const k of CH)PT[k]=j.PT[k];HSLA.set(j.HSLA);Object.assign(WH,j.WH);if(j.WS){for(const w in j.WS)Object.assign(WS[w],j.WS[w]);drawWheels();[['wlLift','lift'],['wlGam','gam'],['wlGain','gain']].forEach(([id,w])=>$('#'+id).value=WS[w].l)}for(const k in DEF){ins[k].value=S[k];sync(k)}updateCurves();drawCrv();hb.onchange()}
function snap(){const s=state();if(HI[hi]===s)return;HI.length=hi+1;HI.push(s);hi=HI.length-1;hist.max=hi;hist.value=hi;HT.length=Math.min(HT.length,hi);const j=hi;setTimeout(()=>{HT[j]=thumb();renderFilm()},140)}
hist.oninput=()=>{hi=+hist.value;applyState(JSON.parse(HI[hi]))};
$('#panel').addEventListener('change',snap);$('#panel').addEventListener('click',e=>{if(e.target.closest('.btn'))setTimeout(snap,60)});cc.addEventListener('pointerup',snap);
// first-run tour
const TOUR=['Twist the lens ring at the bottom to choose a tool.','Pick a setting, then drag anywhere on the photo to change it. Start the drag downward for fine control.','Tap the photo to close the tool. Then swipe for looks, hold to compare, pinch to zoom.'];
function tour(){try{if(localStorage.getItem('lt'))return}catch{}if($('#tour'))return;let i=0;const t=document.createElement('div');t.id='tour';t.className='glass';document.body.append(t);
 (function s(){t.innerHTML='<p>'+TOUR[i]+'</p><button class="btn">'+(i<2?'Next':'Start')+'</button>';t.querySelector('button').onclick=()=>{if(++i<3)s();else{t.remove();try{localStorage.setItem('lt',1)}catch{}}}})()}
if('serviceWorker' in navigator&&location.protocol.startsWith('http'))navigator.serviceWorker.register('sw.js').catch(()=>{});

// ---------- neural depth + virtual light (transformers.js, Depth Anything small; needs internet on first run) ----------
let pipe=null,srcURL='';const depTex=tex2D(6);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
$('#depth').onclick=async()=>{const st=$('#dst');if(!srcURL){st.textContent='Open a photo first';return}
 try{st.textContent='Loading depth model (first time is slow)...';
  if(!pipe){const T=await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2');try{T.env.backends.onnx.wasm.numThreads=2}catch(e){}try{pipe=await T.pipeline('depth-estimation','onnx-community/depth-anything-v2-small',{device:'webgpu',dtype:'fp16'})}catch(e){pipe=await T.pipeline('depth-estimation','onnx-community/depth-anything-v2-small',{device:'wasm',dtype:'q8'})}}
  const d=(await pipe(small(srcIm,518).toDataURL('image/jpeg',.9))).depth,n=d.width*d.height,a=new Uint8Array(n*4);for(let i=0;i<n;i++){const v=d.data[i*d.channels];a[i*4]=a[i*4+1]=a[i*4+2]=v;a[i*4+3]=255}
  gl.activeTexture(gl.TEXTURE6);gl.bindTexture(gl.TEXTURE_2D,depTex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,d.width,d.height,0,gl.RGBA,gl.UNSIGNED_BYTE,a);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
  S.relOn=1;S.lint=ins.lint.value=S.lint||1;sync('lint');st.textContent='Depth ready. Move the light with the sliders.';draw()}
 catch(e){st.textContent='Depth model could not load. It needs internet and a normal page (blocked in the hosted preview).'}};
// ---------- multi-shot stack: average = long exposure, median = remove moving people/cars (use tripod shots) ----------
$('#stackIn').onchange=async e=>{const fs=[...e.target.files],st=$('#sst');if(fs.length<2){st.textContent='Pick at least 2 photos';return}
 st.textContent='Combining...';const bm=await Promise.all(fs.map(f=>createImageBitmap(f,{imageOrientation:'from-image'})));
 const sc=Math.min(1,1600/Math.max(bm[0].width,bm[0].height)),w=Math.round(bm[0].width*sc),h=Math.round(bm[0].height*sc),cx=document.createElement('canvas');cx.width=w;cx.height=h;
 const g=cx.getContext('2d',{willReadFrequently:true}),D=bm.map(b=>{g.drawImage(b,0,0,w,h);return g.getImageData(0,0,w,h).data}),out=g.createImageData(w,h),med=$('#stackMode').value=='median',n=D.length,t=new Array(n);
 for(let i=0;i<w*h*4;i++){if(i%4==3){out.data[i]=255;continue}if(med){for(let k=0;k<n;k++)t[k]=D[k][i];t.sort((a,b)=>a-b);out.data[i]=t[n>>1]}else{let s=0;for(let k=0;k<n;k++)s+=D[k][i];out.data[i]=s/n}}
 g.putImageData(out,0,0);cx.toBlob(b=>$('#file').onchange({target:{files:[b]}}));st.textContent=n+' photos combined'};
// ---------- glass samples the grade's colour + cursor glint ----------
setInterval(()=>{if(!hasImg||document.hidden||liveOn)return;document.body.style.setProperty('--tint',meanRGB(cv).map(v=>Math.round(v*255)).join(','));edgeGlow()},2500);
document.querySelectorAll('.glass').forEach(el=>el.addEventListener('pointermove',ev=>{const r=el.getBoundingClientRect();el.style.setProperty('--mx',ev.clientX-r.left+'px');el.style.setProperty('--my',ev.clientY-r.top+'px')}));

// ================= ASTRO: stack (star-aligned sigma clipping) + gradient sampling =================
const BGC=new Float32Array(12);let srcC=null;
function stars(d,w,h){const L=new Float32Array(w*h);let s1=0,s2=0;for(let i=0;i<w*h;i++){const v=.3*d[i*4]+.59*d[i*4+1]+.11*d[i*4+2];L[i]=v;s1+=v;s2+=v*v}
 const m=s1/(w*h),thr=m+5*Math.sqrt(Math.max(1,s2/(w*h)-m*m)),s=[];
 for(let y=3;y<h-3;y++)for(let x=3;x<w-3;x++){const v=L[y*w+x];if(v<thr)continue;let ok=1;for(let j=-3;j<=3&&ok;j++)for(let i=-3;i<=3;i++)if(L[(y+j)*w+x+i]>v){ok=0;break}if(ok)s.push([x,y,v])}
 return s.sort((a,b)=>b[2]-a[2]).slice(0,40)}
$('#astroIn').onchange=async e=>{const fs=[...e.target.files],st=$('#ast');if(fs.length<2){st.textContent='Pick 2 or more light frames';return}st.textContent='Loading...';
 const bm=await Promise.all(fs.map(f=>createImageBitmap(f,{imageOrientation:'from-image'}))),sc=Math.min(1,1200/Math.max(bm[0].width,bm[0].height)),w=Math.round(bm[0].width*sc),h=Math.round(bm[0].height*sc),c=document.createElement('canvas');c.width=w;c.height=h;
 const g=c.getContext('2d',{willReadFrequently:true}),get=b=>{g.drawImage(b,0,0,w,h);return g.getImageData(0,0,w,h).data};let dk=null;const df=$('#darkIn').files[0];if(df)dk=get(await createImageBitmap(df));
 const F=bm.map(get);if(dk)F.forEach(d=>{for(let i=0;i<d.length;i++)if(i%4<3)d[i]=Math.max(0,d[i]-dk[i])});
 const S0=stars(F[0],w,h),off=F.map((d,fi)=>{if(!fi)return[0,0];const V={};let best=0,bo=[0,0];for(const a of stars(d,w,h))for(const b of S0){const dx=Math.round((b[0]-a[0])/2),dy=Math.round((b[1]-a[1])/2),k=dx+','+dy,n=(V[k]||0)+1;V[k]=n;if(n>best){best=n;bo=[dx*2,dy*2]}}return bo});
 st.textContent='Stacking...';await new Promise(r=>setTimeout(r,30));const out=g.createImageData(w,h),n=F.length;
 for(let y=0;y<h;y++)for(let x=0;x<w;x++)for(let k=0;k<3;k++){const v=[];for(let f=0;f<n;f++){const sx=x-off[f][0],sy=y-off[f][1];if(sx>=0&&sx<w&&sy>=0&&sy<h)v.push(F[f][(sy*w+sx)*4+k])}
  const m=v.reduce((a,b)=>a+b,0)/v.length,sd=Math.sqrt(v.reduce((a,b)=>a+(b-m)**2,0)/v.length)||1,kp=v.filter(q=>Math.abs(q-m)<=2*sd);out.data[(y*w+x)*4+k]=kp.length?kp.reduce((a,b)=>a+b,0)/kp.length:m;out.data[(y*w+x)*4+3]=255}
 g.putImageData(out,0,0);c.toBlob(b=>$('#file').onchange({target:{files:[b]}}));st.textContent=n+' frames stacked (shift alignment only; no rotation correction)'};
$('#bgs').onclick=()=>{if(!srcIm)return;const c=document.createElement('canvas');c.width=c.height=64;const g=c.getContext('2d',{willReadFrequently:true});g.drawImage(srcIm,0,0,64,64);
 const m=(x,y)=>{const d=g.getImageData(x,y,16,16).data;let r=0,gg=0,b=0;for(let i=0;i<d.length;i+=4){r+=d[i];gg+=d[i+1];b+=d[i+2]}const n=256*255;return[(r/n)**2.2,(gg/n)**2.2,(b/n)**2.2]};
 [m(0,48),m(48,48),m(0,0),m(48,0)].forEach((v,i)=>BGC.set(v.map(q=>q*.9),i*3));S.grad=ins.grad.value=1;sync('grad');draw()};
// ================= HEAL BRUSH (classical patch-match clone, runs on device; not generative AI) =================
const hl=$('#hl'),hc=hl.getContext('2d');let hp=0,bb=null;
const inHeal=()=>document.body.dataset.tool==='heal'&&hasImg,hpos=e=>{const r=cv.getBoundingClientRect();return[(e.clientX-r.left)/r.width*W,(e.clientY-r.top)/r.height*H,r.width]};
function hstamp(x,y,rw){const r=$('#hr').value*W/rw,gr=hc.createRadialGradient(x,y,0,x,y,r);gr.addColorStop(0,'rgba(255,40,110,.9)');gr.addColorStop(.7,'rgba(255,40,110,.6)');gr.addColorStop(1,'rgba(255,40,110,0)');hc.fillStyle=gr;hc.fillRect(x-r,y-r,2*r,2*r);bb=[Math.min(bb[0],x-r),Math.min(bb[1],y-r),Math.max(bb[2],x+r),Math.max(bb[3],y+r)]}
function heal(x0,y0,x1,y1){const g=srcC.getContext('2d'),P=8;x0=Math.max(P,x0|0);y0=Math.max(P,y0|0);x1=Math.min(W-P,x1|0);y1=Math.min(H-P,y1|0);const w=x1-x0,h=y1-y0;if(w<2||h<2||w>500||h>500){hc.clearRect(0,0,W,H);return}
 const A=hc.getImageData(x0,y0,w,h).data,I=g.getImageData(0,0,W,H),D=I.data,px=(x,y)=>(y*W+x)*4;let best=1e30,bo=null;
 for(let t=0;t<150;t++){const dx=Math.round((Math.random()*2-1)*(w*2+60)),dy=Math.round((Math.random()*2-1)*(h*2+60));if(Math.abs(dx)<w&&Math.abs(dy)<h)continue;if(x0+dx<P||y0+dy<P||x1+dx>W-P||y1+dy>H-P)continue;
  let c=0;for(let y=y0-6;y<y1+6;y+=2)for(let x=x0-6;x<x1+6;x+=2){if(x>=x0&&x<x1&&y>=y0&&y<y1)continue;const a=px(x,y),b=px(x+dx,y+dy);c+=(D[a]-D[b])**2+(D[a+1]-D[b+1])**2+(D[a+2]-D[b+2])**2}if(c<best){best=c;bo=[dx,dy]}}
 if(bo){for(let y=0;y<h;y++)for(let x=0;x<w;x++){const m=A[(y*w+x)*4+3]/255;if(!m)continue;const a=px(x0+x,y0+y),b=px(x0+x+bo[0],y0+y+bo[1]);for(let k=0;k<3;k++)D[a+k]=D[a+k]*(1-m)+D[b+k]*m}
  g.putImageData(I,0,0);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,imgTex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,srcC);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);draw()}
 hc.clearRect(0,0,W,H)}
cv.addEventListener('pointerdown',e=>{if(!inHeal())return;e.stopImmediatePropagation();if(!srcC){srcC=document.createElement('canvas');srcC.width=W;srcC.height=H;srcC.getContext('2d',{willReadFrequently:true}).drawImage(srcIm,0,0,W,H)}
 hl.width=W;hl.height=H;const r=cv.getBoundingClientRect();Object.assign(hl.style,{left:r.left+'px',top:r.top+'px',width:r.width+'px',height:r.height+'px'});const[x,y,rw]=hpos(e);bb=[x,y,x,y];hp=1;hstamp(x,y,rw)},true);
cv.addEventListener('pointermove',e=>{if(!inHeal())return;e.stopImmediatePropagation();if(hp){const[x,y,rw]=hpos(e);hstamp(x,y,rw)}},true);
cv.addEventListener('pointerup',e=>{if(!inHeal())return;e.stopImmediatePropagation();if(hp){hp=0;heal(...bb)}},true);

// ================= FILM LAB UI, rebates, defects =================
$('#labOn').onchange=e=>{S.lab=+e.target.checked;draw()};$('#negStock').onchange=e=>{S.negS=+e.target.value;draw()};
$('#prtStock').onchange=e=>{S.prtS=+e.target.value;draw()};$('#labView').onchange=e=>{S.labView=+e.target.value;draw()};
$('#rebFmt').onchange=e=>{S.rebFmt=e.target.value;draw()};let animT=0;
$('#anim').onchange=e=>{S.anim=e.target.checked;clearInterval(animT);if(S.anim)animT=setInterval(()=>{if(hasImg&&!document.hidden)draw()},42)};
const REB={'35':{p:4,s:2},tech:{p:2,s:2},'16':{p:1,s:1},'65':{p:5,s:2}},rng=a=>()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296};
function bands(x,w,h,b,f){const R=REB[f],r=rng(7),ng=S.lab&&S.labView==1,bg=ng?'#2b1608':'#0b0b0c',ink=ng?'#e0903a':'#d9d9d9',pw=w/R.p,hw=pw*(f=='65'?.42:.3),hh=b*(f=='16'?.5:.44),code=[['KODAK','5219'],['KODAK','5207'],['FUJI','F-64D'],['EASTMAN','5222']][S.negS||0];
 const ys=R.s==2?[0,h-b]:[0];ys.forEach((y0,si)=>{x.fillStyle=bg;x.fillRect(0,y0,w,b);
  for(let i=0;i<R.p;i++){const cx=pw*(i+.5)+(r()-.5)*b*.03,cy=y0+b*.5+(r()-.5)*b*.03;x.fillStyle='#f2ead2';x.beginPath();x.roundRect(cx-hw/2,cy-hh/2,hw,hh,Math.min(hw,hh)*.28);x.fill();
   x.strokeStyle='rgba(0,0,0,.35)';x.beginPath();x.moveTo(cx-hw/2+r()*hw,cy-hh/2);x.lineTo(cx-hw/2+r()*hw,cy+hh/2);x.stroke()}
  if(si==R.s-1||R.s==1){const ty=R.s==2?y0+b*.5:y0+b*.82;x.fillStyle=ink;x.font=Math.round(b*.2)+'px monospace';x.fillText(code[0]+' '+code[1]+'   '+(10+Math.floor(r()*80))+'A  >',w*.04,y0+b*.88);
   for(let i=0;i<30;i++){x.fillRect(w*.62+i*b*.07,y0+b*.62,Math.max(1,r()*b*.05),b*.2)}x.fillText('KJ 23 '+(1000+Math.floor(r()*8999))+'+'+Math.floor(r()*16),w*.62,y0+b*.5)}})}
function rbUpdate(){const o=$('#rb'),f=S.rebFmt;if(!f||!hasImg){o.style.display='none';return}const R=REB[f],b=Math.max(24,Math.round(H*.075)),ch=H+(R.s==2?2:1)*b,r=cv.getBoundingClientRect(),s=r.width/W;
 o.width=W;o.height=ch;Object.assign(o.style,{display:'block',left:r.left+'px',top:(r.top-b*s)+'px',width:r.width+'px',height:ch*s+'px'});const x=o.getContext('2d');x.clearRect(0,0,W,ch);bands(x,W,ch,b,f)}
$('#exPng').onclick=()=>{if(!hasImg)return;draw();const f=S.rebFmt||($('#reb').checked?'35':'');if(!f)return cv.toBlob(b=>dl(b,'lumina.png'));
 const R=REB[f],b=Math.max(24,Math.round(H*.075)),ch=H+(R.s==2?2:1)*b,o=document.createElement('canvas');o.width=W;o.height=ch;const x=o.getContext('2d');x.fillStyle='#0b0b0c';x.fillRect(0,0,W,ch);x.drawImage(cv,0,b);bands(x,W,ch,b,f);o.toBlob(q=>dl(q,fname('_rebate.png')))};

// ================= PRO TOOLS: zones, hue radar, era, palette, storyboard =================
const ZA=new Float32Array(11);let lastMask=null,pickArm=0;
document.querySelectorAll('[data-sh]').forEach(b=>b.onclick=()=>{S.smear=ins.smear.value=+b.dataset.sh;sync('smear');draw()});
// Zone System
const zc=$('#zchips');['0','I','II','III','IV','V','VI','VII','VIII','IX','X'].forEach((n,i)=>{const b=document.createElement('button');b.className='btn';b.textContent=n;b.onclick=()=>{S.zsel=i;$('#zs').value=ZA[i];S.zhud=1;$('#zhud').checked=true;draw()};zc.append(b)});
$('#zs').oninput=e=>{if(S.zsel>=0){ZA[S.zsel]=+e.target.value;draw()}};$('#zhud').onchange=e=>{S.zhud=+e.target.checked;draw()};
// Hue-angle radar
const rd=$('#radar'),rx=rd.getContext('2d');
function drawRadar(){rx.clearRect(0,0,160,160);const C=80,R=68;if(rx.createConicGradient){const g=rx.createConicGradient(-1.5708,C,C);for(let i=0;i<=12;i++)g.addColorStop(i/12,'hsl('+(i*30)+',80%,55%)');rx.strokeStyle=g}else rx.strokeStyle='#888';
 rx.lineWidth=14;rx.beginPath();rx.arc(C,C,R,0,7);rx.stroke();
 if(S.hsOn){const a0=S.hsHue*6.2832-1.5708,w=(S.hsW||.05)*6.2832,a1=a0+(S.hsShift||0)*.25*6.2832,rr=R*(.55+.35*(S.hsSat||0));rx.fillStyle='rgba(255,255,255,.25)';rx.beginPath();rx.moveTo(C,C);rx.arc(C,C,R,a0-w,a0+w);rx.fill();rx.fillStyle='#fff';rx.beginPath();rx.arc(C+Math.cos(a1)*rr,C+Math.sin(a1)*rr,7,0,7);rx.fill()}}
rd.onpointerdown=rd.onpointermove=e=>{if(!S.hsOn||(e.type=='pointermove'&&!e.buttons))return;const r=rd.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*160-80,y=(e.clientY-r.top)/r.height*160-80,a0=S.hsHue*6.2832-1.5708;let d=Math.atan2(y,x)-a0;d=Math.atan2(Math.sin(d),Math.cos(d));
 S.hsShift=Math.max(-1,Math.min(1,d/6.2832/.25));S.hsSat=Math.max(-1,Math.min(1,(Math.hypot(x,y)/68-.55)/.35));['hsShift','hsSat'].forEach(k=>{ins[k].value=S[k];sync(k)});drawRadar();draw()};
$('#pick').onclick=()=>{pickArm=1;setTool('');toast('Tap a colour in the photo')};
$('#hsOff').onclick=()=>{S.hsOn=0;draw();drawRadar()};
function pickAt(e){pickArm=0;const r=cv.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)/r.width*W),y=Math.floor((e.clientY-r.top)/r.height*H),p=new Uint8Array(4);gl.readPixels(x,H-1-y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);
 const m=Math.max(p[0],p[1],p[2]),n=Math.min(p[0],p[1],p[2]),d=m-n;let hu=0;if(d)hu=m==p[0]?((p[1]-p[2])/d+6)%6:m==p[1]?(p[2]-p[0])/d+2:(p[0]-p[1])/d+4;
 S.hsHue=hu/6;S.hsOn=1;S.hsW=S.hsW||.08;ins.hsW.value=S.hsW;sync('hsW');setTool('hue');drawRadar();draw()}
drawRadar();
// Time machine
const ERAS=[[1920,{mono:1,con:.35,grain:.8,vig:.45,halo:.1,scr:.7,dust:.6,gwv:.5,hi:-.2,clar:-.2}],[1942,{sat:.65,con:.25,temp:.1,dyeY:.3,halo:.15,grain:.3,vig:.15}],[1965,{temp:.3,sat:.35,con:.25,vig:.25,grain:.25,dust:.2}],[1975,{temp:.25,sat:-.1,con:.15,grain:.7,vig:.35,halo:.2,dust:.2}],[1985,{sat:.2,con:.15,tint:.1,grain:.45,halo:.15,vig:.2}],[1998,{sat:-.1,con:.12,sh:.2,temp:-.1,grain:.3,halo:.3}],[2025,{}]];
function eraAt(y){let i=0;while(i<ERAS.length-2&&y>ERAS[i+1][0])i++;const[y0,a]=ERAS[i],[y1,b]=ERAS[i+1],t=Math.max(0,Math.min(1,(y-y0)/(y1-y0))),o={};for(const k of new Set([...Object.keys(a),...Object.keys(b)]))o[k]=Math.max(RANGE[k][0],Math.min(RANGE[k][1],(a[k]||0)*(1-t)+(b[k]||0)*t));return o}
$('#era').oninput=e=>{$('#eraY').textContent=e.target.value;setAll(eraAt(+e.target.value))};
// Palette transfer (per-channel histogram matching, baked into a 1D table)
const PTX=tex2D(7);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
const histo=src=>{const c=document.createElement('canvas');c.width=c.height=128;const g=c.getContext('2d',{willReadFrequently:true});g.drawImage(src,0,0,128,128);const d=g.getImageData(0,0,128,128).data,h=[0,1,2].map(()=>new Float32Array(256));for(let i=0;i<d.length;i+=4)for(let k=0;k<3;k++)h[k][d[i+k]]++;return{h,d}};
const cdf=h=>{let s=0;const c=new Float32Array(256);for(let i=0;i<256;i++){s+=h[i];c[i]=s}for(let i=0;i<256;i++)c[i]/=s;return c};
$('#palIn').onchange=async e=>{if(!hasImg||!e.target.files[0])return;const ref=await createImageBitmap(e.target.files[0]);S.ptOn=0;draw();const A=histo(cv),B=histo(ref),out=new Uint8Array(1024);
 for(let k=0;k<3;k++){const ca=cdf(A.h[k]),cb=cdf(B.h[k]);let j=0;for(let i=0;i<256;i++){while(j<255&&cb[j]<ca[i])j++;out[i*4+k]=j}}for(let i=0;i<256;i++)out[i*4+3]=255;
 gl.activeTexture(gl.TEXTURE7);gl.bindTexture(gl.TEXTURE_2D,PTX);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,256,1,0,gl.RGBA,gl.UNSIGNED_BYTE,out);S.ptOn=1;S.ptAmt=ins.ptAmt.value=1;sync('ptAmt');
 const q={};for(let i=0;i<B.d.length;i+=4){const k=(B.d[i]>>6)*16+(B.d[i+1]>>6)*4+(B.d[i+2]>>6);q[k]=(q[k]||0)+1}
 $('#pal').innerHTML=Object.entries(q).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([k])=>{k=+k;return'<i style="background:rgb('+(k>>4)*85+','+((k>>2)&3)*85+','+(k&3)*85+')"></i>'}).join('');draw()};
// One-photo storyboard
$('#board').onclick=()=>{if(!hasImg)return;draw();let b={x0:.3,y0:.25,x1:.7,y1:.75},f={x0:.4,y0:.3,x1:.6,y1:.5};
 if(lastMask){const{w,h,a}=lastMask,bx=c=>{let x0=w,y0=h,x1=0,y1=0;for(let y=0;y<h;y++)for(let x=0;x<w;x++)if(c(a[y*w+x])){x0=Math.min(x0,x);y0=Math.min(y0,y);x1=Math.max(x1,x);y1=Math.max(y1,y)}return x1>x0?{x0:x0/w,y0:y0/h,x1:x1/w,y1:y1/h}:null};
  b=bx(v=>v>0)||b;f=bx(v=>v==3)||{x0:b.x0+(b.x1-b.x0)*.3,x1:b.x1-(b.x1-b.x0)*.3,y0:b.y0,y1:b.y0+(b.y1-b.y0)*.25}}
 const box=(r,s)=>{const cx=(r.x0+r.x1)/2,cy=(r.y0+r.y1)/2;let ww=Math.min(1,Math.max((r.x1-r.x0)*s,(r.y1-r.y0)*s*H/W,.12)),hh=ww*W/H;if(hh>1){hh=1;ww=H/W}return[Math.max(0,Math.min(1-ww,cx-ww/2))*W,Math.max(0,Math.min(1-hh,cy-hh/2))*H,ww*W,hh*H]};
 const shots=[['1  WIDE',[0,0,W,H]],['2  MEDIUM',box(b,1.5)],['3  CLOSE-UP',box({x0:b.x0,x1:b.x1,y0:b.y0,y1:b.y0+(b.y1-b.y0)*.5},1.3)],['4  EXTREME CLOSE-UP',box(f,1.8)]];
 const tw=640,th=Math.round(tw*H/W),p=24,o=document.createElement('canvas');o.width=tw*2+p*3;o.height=(th+p+30)*2+p;const x=o.getContext('2d');x.fillStyle='#0b0b0b';x.fillRect(0,0,o.width,o.height);
 shots.forEach(([n,r],i)=>{const cx=p+(i%2)*(tw+p),cy=p+Math.floor(i/2)*(th+p+30);x.drawImage(cv,r[0],r[1],r[2],r[3],cx,cy,tw,th);x.strokeStyle='#fff';x.strokeRect(cx,cy,tw,th);x.fillStyle='#fff';x.font='16px monospace';x.fillText(n,cx,cy+th+22)});
 o.toBlob(q=>dl(q,'storyboard.png'))};


// ================= UI v4 "Iris": a lens-barrel ring is the whole interface =================
const TOOLS=[['light','Light'],['zone','Zones'],['curves','Curves'],['color','Color'],['mixer','Mixer'],['hue','Hue'],['palette','Palette'],['cinema','Cinema'],['masters','Masters'],['film','Film'],['era','Era'],['lab','Lab'],['pro','Pro'],['luts','LUTs'],['lens','Lens'],['heal','Heal'],['masks','Masks'],['relight','Relight'],['gobo','Gobo'],['astro','Astro'],['stack','Stack'],['board','Board'],['frame','Frame'],['export','Export']],N=TOOLS.length,STEP=360/N;
const ui=document.createElement('div');ui.id='ui';ui.innerHTML='<div id="read" hidden><b></b><i></i></div><div id="chips"></div><div id="barrel"></div><span id="dot"></span>';document.body.append(ui);
const barrel=$('#barrel'),chips=$('#chips'),read=$('#read');
TOOLS.forEach(([k,n],i)=>{const e=document.createElement('em');e.textContent=n;e.style.transform='rotate('+i*STEP+'deg) translateY(calc(var(--R) * -1 + 46px)) translateX(-50%)';barrel.append(e)});
const IC={scope:'<svg viewBox="0 0 24 24"><path d="M3 12h3l2-6 4 12 3-9 2 3h4"/></svg>',open:'<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>',cmp:'<svg viewBox="0 0 24 24"><path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>'};
$('#scopeBtn').innerHTML=IC.scope;$('#open2').innerHTML=IC.open;
const cmp=document.createElement('button');cmp.className='rb';cmp.title='Hold to see the original';cmp.innerHTML=IC.cmp;$('#scopeBtn').before(cmp);
cmp.onpointerdown=()=>{S.orig=1;draw()};cmp.onpointerup=cmp.onpointerleave=()=>{if(S.orig){S.orig=0;draw()}};
let rot=0,idx=0,dr=null,act=null,sc=null;
const curIdx=()=>((Math.round(-rot/STEP)%N)+N)%N,place=(r,a)=>{barrel.style.transition=a?'transform .6s cubic-bezier(.2,1.25,.4,1)':'none';barrel.style.transform='rotate('+r+'deg)'};
const lit=i=>[...barrel.children].forEach((e,j)=>e.classList.toggle('on',j===i));
barrel.onpointerdown=e=>{barrel.setPointerCapture(e.pointerId);const b=barrel.getBoundingClientRect(),cx=b.left+b.width/2,cy=b.top+b.height/2;dr={cx,cy,a:Math.atan2(e.clientY-cy,e.clientX-cx),r0:rot,m:0}};
barrel.onpointermove=e=>{if(!dr)return;let d=(Math.atan2(e.clientY-dr.cy,e.clientX-dr.cx)-dr.a)*57.2958;d=((d+540)%360)-180;rot=dr.r0+d;dr.m=Math.max(dr.m,Math.abs(d));place(rot,0);const i=curIdx();if(i!==idx){idx=i;lit(i);navigator.vibrate&&navigator.vibrate(4)}};
barrel.onpointerup=()=>{if(!dr)return;rot=Math.round(rot/STEP)*STEP;idx=curIdx();place(rot,1);lit(idx);const t=TOOLS[idx][0];dr=null;setTool(t)};
const showVal=()=>{if(act)read.querySelector('b').textContent=+act.step>=1?act.value:(+act.value).toFixed(2)};
function pick(i,b,n){act=i;chips.querySelectorAll('.chip').forEach(c=>c.classList.toggle('on',c===b));read.hidden=false;read.querySelector('i').textContent=n;showVal();b.scrollIntoView({inline:'center',block:'nearest',behavior:'smooth'})}
document.addEventListener('input',e=>{if(e.target===act)showVal()},true);
const _st=setTool;setTool=function(t){if(t){document.body.classList.remove('loupe','wipe');document.querySelectorAll('#side .rb').forEach(x=>x.classList.remove('on'))}_st(t);if(t)perfGo();guideUpdate();document.body.classList.toggle('open',!!t);chips.innerHTML='';act=null;read.hidden=true;
 const i=TOOLS.findIndex(x=>x[0]===t);if(i>=0&&i!==curIdx()){const d=((i-curIdx()+N/2+N)%N)-N/2;rot-=d*STEP;idx=i;place(rot,1)}lit(i);if(!t)return;
 const secs=[...document.querySelectorAll('#panel [data-t="'+t+'"]')],rows=secs.flatMap(s=>[...s.querySelectorAll('.row')]).filter(r=>r.querySelector('input[type=range]'));
 rows.forEach(r=>{const n=(r.querySelector('span')||{}).textContent||'',b=document.createElement('button');b.className='chip';b.textContent=n;b.onclick=()=>pick(r.querySelector('input'),b,n);chips.append(b)});
 if(rows.length)pick(rows[0].querySelector('input'),chips.firstChild,chips.firstChild.textContent);
 $('#panel').hidden=!secs.some(s=>[...s.children].some(c=>!c.matches('.row,summary')))};
// the whole photo is the slider while a tool is open
cv.addEventListener('pointerdown',e=>{const t=document.body.dataset.tool;if(!t||!act||t==='heal'||/loupe|wipe/.test(document.body.className)||mDraw)return;e.stopImmediatePropagation();cv.setPointerCapture(e.pointerId);sc={x:e.clientX,y:e.clientY,v:+act.value,m:0}},true);
cv.addEventListener('pointermove',e=>{if(!sc)return;e.stopImmediatePropagation();const dx=e.clientX-sc.x,dy=e.clientY-sc.y;sc.m=Math.max(sc.m,Math.hypot(dx,dy));if(sc.m<8)return;document.body.classList.add('scrub');
 const f=sc.y0===undefined&&dy>50?.1:1,rg=act.max-act.min;act.value=Math.max(+act.min,Math.min(+act.max,sc.v+dx/(innerWidth*.8)*rg*f));act.dispatchEvent(new Event('input',{bubbles:true}));
 if(+act.value===0&&!sc.z&&navigator.vibrate){navigator.vibrate(6);sc.z=1}else if(+act.value!==0)sc.z=0},true);
cv.addEventListener('pointerup',e=>{if(!sc)return;e.stopImmediatePropagation();document.body.classList.remove('scrub');if(sc.m<8)setTool('');else act.dispatchEvent(new Event('change',{bubbles:true}));sc=null},true);
addEventListener('deviceorientation',e=>{const s=document.body.style;s.setProperty('--gx',(50+Math.max(-45,Math.min(45,e.gamma||0)))+'%')});
setTool('');

// ================= START SCREEN: every look drifting past as two film reels (rendered by the real shader) =================
function heroStrips(){const keep=S,ow=W,oh=H,tw=200,th=112,ch=cv.height,z=Object.fromEntries(Object.keys(DEF).map(k=>[k,0])),items=[];
 gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,imgTex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,sampleCanvas());gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
 W=1600;H=900;
 for(const d in DECKS)for(const n in DECKS[d]){S={...z,...DECKS[d][n],lutOn:0,lutOn2:0,lutMix:1,lutMix2:1,maskOn:0,neg:0,zone:-1,orig:0};gl.viewport(0,0,tw,th);setU();gl.drawArrays(gl.TRIANGLE_STRIP,0,4);
  const k=document.createElement('canvas');k.width=tw;k.height=th;k.getContext('2d').drawImage(cv,0,ch-th,tw,th,0,0,tw,th);items.push([n,d,k.toDataURL('image/jpeg',.7)])}
 S=keep;W=ow;H=oh;window.HERO=items;const reel=document.createElement('div');reel.id='reel';
 [0,1].forEach(r=>{const row=document.createElement('div');row.className='lane l'+r;const list=items.filter((_,i)=>i%2==r);
  row.innerHTML=list.concat(list).map(([n,d,u])=>'<button data-n="'+n+'" data-d="'+d+'"><img src="'+u+'" alt=""><span>'+n+'</span></button>').join('');reel.append(row)});
 $('#empty').prepend(reel);
 reel.onclick=async e=>{const b=e.target.closest('button');if(!b)return;deck=b.dataset.d;cp=DECKS[deck][b.dataset.n];await $('#sample').onclick();renderDeck();applyCP();slate('Look: '+b.dataset.n)}}
try{heroStrips()}catch(e){}
// ================= LIVE CINEMA MONITOR: the phone camera, graded in real time =================
let vid=null,liveOn=0,fr=0,lf=0;
async function live(on){if(!on){liveOn=0;if(vid){vid.srcObject.getTracks().forEach(t=>t.stop());vid=null}document.body.classList.remove('live');S.anim=false;cv.style.scale='';if(WL){try{WL.release()}catch(e){}WL=null}return}
 try{const st=await camStream();vid=document.createElement('video');vid.muted=true;vid.playsInline=true;vid.srcObject=st;await vid.play();
  W=vid.videoWidth;H=vid.videoHeight;cv.width=W;cv.height=H;if(!hasImg){hasImg=true;document.body.classList.add('has');if($('#empty'))$('#empty').remove();setTool('light')}
  srcIm=null;liveOn=1;S.anim=true;try{WL=await navigator.wakeLock.request('screen')}catch(e){}document.body.classList.add('live');try{const fm=st.getVideoTracks()[0].getSettings().facingMode;cv.style.scale=(fm==='user'||!fm)?'-1 1':''}catch(e){}$('#lfps').textContent=W+'x'+H;fit();
  (function f(now){if(!liveOn)return;requestAnimationFrame(f);if(now-lf<32||vid.readyState<2)return;lf=now;gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,imgTex);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,gl.RGBA,gl.UNSIGNED_BYTE,vid);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);draw();if(document.body.classList.contains('sc')&&++fr%8==0)scope()})()}
 catch(e){toast('Camera not available here')}}
const camB=document.createElement('button');camB.className='rb';camB.title='Live monitor';camB.innerHTML='<svg viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>';cmp.before(camB);
camB.onclick=()=>live(!liveOn);$('#liveBtn').onclick=()=>live(true);
const sh=document.createElement('button');sh.id='shutter';sh.title='Capture frame';document.body.append(sh);
function doCapture(){const c=document.createElement('canvas');c.width=vid.videoWidth;c.height=vid.videoHeight;c.getContext('2d').drawImage(vid,0,0);c.toBlob(async b=>{await live(false);await $('#file').onchange({target:{files:[b]}});slate('Captured');tixAdd('capture')})}

// ================= UI v5: filmstrip history, glass loupe, look wipe =================
const HT=[];
const thumb=()=>{const t=document.createElement('canvas');t.width=72;t.height=Math.max(1,Math.round(72*H/W));t.getContext('2d').drawImage(cv,0,0,t.width,t.height);return t.toDataURL('image/jpeg',.6)};
const film=document.createElement('div');film.id='film';film.hidden=true;document.body.append(film);
function renderFilm(){film.innerHTML='';if(HI.length<2){film.hidden=true;return}film.hidden=false;
 HI.forEach((_,i)=>{const b=document.createElement('button');b.className=(i===hi?'on':'')+(CIRC.has(i)?' circ':'');b.dataset.t=i+1;if(HT[i])b.style.backgroundImage='url('+HT[i]+')';b.onclick=()=>{hi=i;hist.value=i;applyState(JSON.parse(HI[i]));renderFilm()};let lt;b.onpointerdown=()=>{lt=setTimeout(()=>{CIRC.has(i)?CIRC.delete(i):CIRC.add(i);navigator.vibrate&&navigator.vibrate(12);renderFilm()},450)};b.onpointerup=b.onpointerleave=()=>clearTimeout(lt);film.append(b)});film.lastChild.scrollIntoView({inline:'end',block:'nearest'})}
// glass loupe: a round 100% magnifier follows your finger
const lp=document.createElement('canvas');lp.id='loupe';lp.width=lp.height=160;document.body.append(lp);const lx=lp.getContext('2d'),mode=m=>document.body.classList.contains(m);
const loupeAt=e=>{const r=cv.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*W,y=(e.clientY-r.top)/r.height*H;lx.clearRect(0,0,160,160);lx.drawImage(cv,x-80,y-80,160,160,0,0,160,160);Object.assign(lp.style,{display:'block',left:e.clientX-80+'px',top:e.clientY-210+'px'})};
cv.addEventListener('pointerdown',e=>{if(!mode('loupe'))return;e.stopImmediatePropagation();cv.setPointerCapture(e.pointerId);loupeAt(e)},true);
cv.addEventListener('pointermove',e=>{if(!mode('loupe')||!(e.buttons||e.pressure))return;e.stopImmediatePropagation();loupeAt(e)},true);
cv.addEventListener('pointerup',e=>{if(!mode('loupe'))return;e.stopImmediatePropagation();lp.style.display='none'},true);
// look wipe: drag across the photo to reveal the next look, release past the middle to keep it
const wl=document.createElement('div');wl.id='wl';document.body.append(wl);let wn=null,wx=0;
function wipeDraw(x){const keep=S,ns=Object.keys(DECKS[deck]),i=ns.findIndex(n=>DECKS[deck][n]===cp);wn=ns[(i+1)%ns.length];const z=Object.fromEntries(Object.keys(DEF).map(k=>[k,0])),a=S.anim;
 S.anim=true;draw();S.anim=a;S={...keep,...z,...DECKS[deck][wn],anim:true};gl.enable(gl.SCISSOR_TEST);gl.scissor(0,0,Math.round(x*W),H);gl.viewport(0,0,W,H);setU();gl.drawArrays(gl.TRIANGLE_STRIP,0,4);gl.disable(gl.SCISSOR_TEST);S=keep}
const wp=e=>{const r=cv.getBoundingClientRect();wx=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width));wl.style.display='block';wl.style.left=e.clientX+'px';wipeDraw(wx)};
cv.addEventListener('pointerdown',e=>{if(!mode('wipe'))return;e.stopImmediatePropagation();cv.setPointerCapture(e.pointerId);wp(e)},true);
cv.addEventListener('pointermove',e=>{if(!mode('wipe')||!(e.buttons||e.pressure))return;e.stopImmediatePropagation();wp(e)},true);
cv.addEventListener('pointerup',e=>{if(!mode('wipe'))return;e.stopImmediatePropagation();wl.style.display='none';if(wx>.5&&wn){cp=DECKS[deck][wn];burn();clk();applyCP();slate('Look: '+wn)}else draw()},true);
const side=document.createElement('div');side.id='side';
[['loupe','<svg viewBox="0 0 24 24"><circle cx="11" cy="11" r="6"/><path d="M20 20l-4-4"/></svg>','Touch the photo to magnify it'],['wipe','<svg viewBox="0 0 24 24"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M12 5v14"/></svg>','Drag across the photo to wipe to the next look']].forEach(([m,svg,tip])=>{
 const b=document.createElement('button');b.className='rb';b.title=m;b.innerHTML=svg;b.onclick=()=>{const on=!mode(m);document.body.classList.remove('loupe','wipe');side.querySelectorAll('.rb').forEach(x=>x.classList.remove('on'));if(on){setTool('');document.body.classList.add(m);b.classList.add('on');toast(tip)}};side.append(b)});
document.body.append(side);

// ================= OPENING SEQUENCE =================
const idb=(m,k,v)=>new Promise(res=>{try{const o=indexedDB.open('lumina',1);o.onupgradeneeded=()=>o.result.createObjectStore('s');o.onsuccess=()=>{const s=o.result.transaction('s',m==='get'?'readonly':'readwrite').objectStore('s'),r=m==='get'?s.get(k):s.put(v,k);r.onsuccess=()=>res(r.result);r.onerror=()=>res()};o.onerror=()=>res()}catch(e){res()}});
const thumbBig=()=>{const t=document.createElement('canvas');t.width=240;t.height=Math.max(1,Math.round(240*H/W));t.getContext('2d').drawImage(cv,0,0,t.width,t.height);return t.toDataURL('image/jpeg',.7)};
setInterval(()=>{if(!hasImg||liveOn||document.hidden)return;idb('put','state',{s:state(),t:thumbBig()})},4000);
// safelight "develop" effect whenever a photo opens
const sf=document.createElement('div');sf.id='safe';document.body.append(sf);
function dev(){cv.classList.remove('develop');sf.classList.remove('go');void cv.offsetWidth;cv.classList.add('develop');sf.classList.add('go')}
(function(){const E=$('#empty');if(!E)return;
 // 3. time-of-day light
 const hr=new Date().getHours(),P=hr<5||hr>=21?['night','#0a1030','#000']:hr<7?['dawn','#ffb38a','#2b2f6a']:hr<16?['daylight','#aab6c8','#1b2230']:hr<19?['golden hour','#ff9a3c','#3a1d4a']:['blue hour','#2a4cff','#070a24'];
 const tod=document.createElement('div');tod.id='tod';tod.style.background='linear-gradient(180deg,'+P[1]+','+P[2]+')';E.prepend(tod);
 const g=document.createElement('div');g.id='greet';g.textContent=(hr<5?'good night':hr<12?'good morning':hr<17?'good afternoon':'good evening')+' \u00b7 '+P[0];E.append(g);
 // 2. projector beam: frames are dark until the light passes over them
 const R0=$('#reel');if(R0){R0.style.filter='brightness(.3) saturate(.6)';const L=R0.cloneNode(true);L.id='reelLit';L.style.filter='';R0.after(L);
  [R0,L].forEach(r=>r.querySelectorAll('.lane').forEach(l=>{l.style.animation='none';void l.offsetWidth;l.style.animation=''}))}
 const cone=document.createElement('div');cone.id='cone';E.append(cone);
 let last=0;const bm=(x,y)=>{document.body.style.setProperty('--bx',x+'px');document.body.style.setProperty('--by',y+'px')};bm(innerWidth/2,innerHeight/2);
 addEventListener('pointermove',e=>{last=performance.now();bm(e.clientX,e.clientY)});
 addEventListener('deviceorientation',e=>{if(performance.now()-last>1500)bm(innerWidth*(.5+Math.max(-30,Math.min(30,e.gamma||0))/60),innerHeight*.5)});
 // 7. resume card
 idb('get','state').then(st=>{if(!st||!st.t||!$('#empty'))return;const c=document.createElement('button');c.id='resume';c.innerHTML='<img src="'+st.t+'" alt=""><span>Continue your last edit</span>';($('#mid')||E).append(c);
  c.onclick=async()=>{const src=await idb('get','src');if(!src)return;await $('#file').onchange({target:{files:[src]}});applyState(JSON.parse(st.s));snap();draw()}});
 // 1+4(+5). iris opens on a clapperboard, then it claps shut and cuts to the reels
 if(matchMedia('(prefers-reduced-motion:reduce)').matches)return;
 let take=1;try{take=(+localStorage.getItem('take')||0)+1;localStorage.setItem('take',take)}catch(e){}
 const I=document.createElement('div');I.id='intro';I.innerHTML='<div id="slate"><div id="stick"></div><div id="board"><b>LUMINA STUDIO</b><div><span>SCENE</span>1</div><div><span>TAKE</span><u>'+take+'</u></div><div><span>ROLL</span>A</div><div><span>DATE</span>'+new Date().toLocaleDateString()+'</div></div></div><svg id="irisSvg"><path id="irisP" fill="#000" fill-rule="evenodd"/></svg>';document.body.append(I);
 const sv=$('#irisSvg'),ip=$('#irisP'),w=innerWidth,h=innerHeight,cx=w/2,cy=h/2,Rm=Math.hypot(w,h)/2*1.1;sv.setAttribute('viewBox','0 0 '+w+' '+h);
 const hexa=(r,a)=>[0,1,2,3,4,5].map(i=>{const t=a+i*Math.PI/3;return(i?'L':'M')+(cx+Math.cos(t)*r).toFixed(1)+' '+(cy+Math.sin(t)*r).toFixed(1)}).join('')+'Z',t0=performance.now();let done=0;
 (function f(now){const p=Math.min(1,(now-t0)/1100),e=1-Math.pow(1-p,3);ip.setAttribute('d','M0 0H'+w+'V'+h+'H0Z'+hexa(Rm*e,(1-e)*.9));if(p<1)requestAnimationFrame(f);else ip.setAttribute('d','M0 0')})(t0);
 const clapNow=()=>{if(done)return;done=1;$('#stick').classList.add('hit');navigator.vibrate&&navigator.vibrate([20,40,30]);
  try{if($('#snd').checked){AC=AC||new AudioContext();const b=AC.createBuffer(1,6000,44100),d=b.getChannelData(0);for(let i=0;i<6000;i++)d[i]=(Math.random()*2-1)*Math.exp(-i/700);const s=AC.createBufferSource();s.buffer=b;s.connect(AC.destination);s.start()}}catch(e){}
  setTimeout(()=>{I.classList.add('out');setTimeout(()=>I.remove(),600)},350)};
 I.remove();
 // shake the phone for a new take
 let ls=0;addEventListener('devicemotion',e=>{const a=e.accelerationIncludingGravity;if(!a||!$('#empty'))return;if(Math.hypot(a.x,a.y,a.z)>28&&performance.now()-ls>1500){ls=performance.now();take++;try{localStorage.setItem('take',take)}catch(x){}toast('Take '+take);burn()}});
})();

// ================= CLAPPERBOARD EVERYWHERE, TAKES, SCENE OF THE DAY =================
const CIRC=new Set(),LOOKS=new Set(),lsn=k=>{try{return+localStorage.getItem(k)||0}catch(e){return 0}};
const lookName=()=>{const d=DECKS[deck]||{};for(const n in d)if(d[n]===cp)return n;return'custom'},fname=s=>'S1_T'+Math.max(1,HI.length)+'_'+lookName().replace(/[^\w]+/g,'-')+s;
const ms=document.createElement('div');ms.id='mslate';document.body.append(ms);
function slate(t){LOOKS.add(t);ms.innerHTML='<i></i><b>TAKE '+Math.max(1,HI.length)+'</b> ';ms.append(document.createTextNode(t));ms.classList.remove('go');void ms.offsetWidth;ms.classList.add('go')}
function wrapUp(n){let w=$('#wrap');if(!w){w=document.createElement('div');w.id='wrap';document.body.append(w)}
 w.innerHTML='<svg viewBox="0 0 48 48"><circle cx="24" cy="24" r="20"/><circle cx="24" cy="12" r="4"/><circle cx="24" cy="36" r="4"/><circle cx="12" cy="24" r="4"/><circle cx="36" cy="24" r="4"/></svg><b>that\'s a wrap</b><i>'+n+'</i>';w.classList.remove('go');void w.offsetWidth;w.classList.add('go')}
const perf=document.createElement('div');perf.id='perf';document.body.append(perf);function perfGo(){perf.classList.remove('go');void perf.offsetWidth;perf.classList.add('go')}
// film-leader countdown before the live shutter
const cd=document.createElement('div');cd.id='cd';cd.innerHTML='<b></b>';document.body.append(cd);let cdOn=0;
sh.onclick=()=>{if(cdOn)return;cdOn=1;let n=3;const tick=()=>{if(!n){cd.style.display='none';cdOn=0;doCapture();return}cd.style.display='grid';cd.firstChild.textContent=n--;cd.classList.remove('sw');void cd.offsetWidth;cd.classList.add('sw');navigator.vibrate&&navigator.vibrate(8);setTimeout(tick,800)};tick()};
// circled takes (long-press a thumbnail in the strip), export them, roll credits
$('#expCirc').onclick=async()=>{if(!CIRC.size){toast('Long-press a take in the strip to circle it');return}const keep=hi;
 for(const i of[...CIRC].sort((a,b)=>a-b)){applyState(JSON.parse(HI[i]));draw();await new Promise(r=>cv.toBlob(b=>{dl(b,'S1_T'+(i+1)+'_circled.png');r()}));await new Promise(r=>setTimeout(r,300))}applyState(JSON.parse(HI[keep]));draw()};
function rollCredits(){const c=document.createElement('div');c.id='credits';c.innerHTML='<div><h1>LUMINA STUDIO</h1><p>a film by you</p><p>photos graded<br><b>'+lsn('photos')+'</b></p><p>takes this session<br><b>'+HI.length+'</b></p><p>looks tried<br><b>'+LOOKS.size+'</b></p><p>shot on a phone<br>graded in the browser</p><p>THE END</p></div>';c.onclick=()=>c.remove();document.body.append(c);setTimeout(()=>c.remove(),16000)}
// shake to cut (undo), off by default
let ls2=0;addEventListener('devicemotion',e=>{const a=e.accelerationIncludingGravity;if(!a||!hasImg||!$('#shk').checked)return;if(Math.hypot(a.x,a.y,a.z)>28&&performance.now()-ls2>1500&&hi>0){ls2=performance.now();hi--;hist.value=hi;applyState(JSON.parse(HI[hi]));renderFilm();slate('Cut \u00b7 back to take '+(hi+1))}});
// start screen: actions as lens-style chips at the bottom, scene of the day in the middle
[['open','<svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg>Open'],['sample','<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/></svg>Sample'],['liveBtn','<svg viewBox="0 0 24 24"><path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/></svg>Live']].forEach(([id,h])=>{const b=$('#'+id);if(b)b.innerHTML=h});
(function(){const E=$('#empty');if(!E)return;const SC=[['Rainy neon thriller','Arthouse','Tokyo Neon'],['Last light on a western','Genre','Neo-Western Amber'],['A silent-era memory','Black & White','Ortho Cinema'],['Nordic winter noon','Arthouse','Nordic Cold'],['Sci-fi sodium streets','Genre','Sci-Fi Sodium'],['Kodachrome summer','Vintage','Kodachrome 64'],['Golden hour in Tehran','Arthouse','Iranian Golden Hour'],['Bleach bypass war film','Vintage','Bleach Bypass'],['Emerald thriller','Genre','Thriller Emerald'],['Technicolor musical','Vintage','Technicolor 3-Strip'],['A24 daylight drama','Arthouse','A24 Daylight'],['Tri-X street photographer','Black & White','Tri-X 400 +1'],['Autochrome memory','Vintage','Autochrome'],['Night shoot on 500T','35mm Motion','Vision3 500T']];
 const [t,dk,lk]=SC[Math.floor(Date.now()/864e5)%SC.length],mid=document.createElement('div');mid.id='mid';mid.innerHTML='<div id="scene"><span>scene of the day</span><b>'+t+'</b><button class="chip">grade it</button></div>';E.append(mid);
 mid.querySelector('button').onclick=async()=>{deck=dk;cp=DECKS[dk][lk];await $('#sample').onclick();renderDeck();applyCP();slate('Scene: '+t)}})();

// ================= LOCAL MASKS: Flag (linear), Spot (radial), Range (luminance), Gel (colour), Paint (brush) =================
var mkReady=0,mDraw=null,mg=null;let MASKS=[0,1,2,3].map(()=>({t:0,a:[.5,.5,.5,.5],f:.4,inv:0,ev:0,con:0,sat:0,tmp:0})),mAct=-1,BR=null;
const MA=new Float32Array(16),MB=new Float32Array(16),MC=new Float32Array(16),MN=['','Flag','Spot','Range','Gel','Paint'];
function packMasks(){MASKS.forEach((m,i)=>{MA.set([m.t,m.a[0],m.a[1],m.a[2]],i*4);MB.set([m.a[3],m.f,m.inv,0],i*4);MC.set([m.ev,m.con,m.sat,m.tmp],i*4)})}
const BRT=tex2D(8);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(4));
mg=document.createElement('canvas');mg.id='mg';document.body.append(mg);
const MF=[['mEv','ev'],['mCon','con'],['mSat','sat'],['mTmp','tmp'],['mF','f']];
function renderSlots(){renderRail();const s=$('#mslots');s.innerHTML='';MASKS.forEach((m,i)=>{if(!m.t)return;const b=document.createElement('button');b.className='btn'+(i===mAct?' on':'');b.textContent=(i+1)+' '+MN[m.t];b.onclick=()=>{mAct=i;syncMask();draw()};s.append(b)})}
function syncMask(){renderSlots();const m=MASKS[mAct];if(m&&m.t)MF.forEach(([k,f])=>{ins[k].value=m[f];S[k]=m[f];sync(k)});guideUpdate()}
MF.forEach(([k,f])=>ins[k].addEventListener('input',e=>{if(mAct>=0&&MASKS[mAct].t){MASKS[mAct][f]=+e.target.value;draw()}}));
const mpos=e=>{const r=cv.getBoundingClientRect();return[(e.clientX-r.left)/r.width,1-(e.clientY-r.top)/r.height]};
function clearCh(i){if(BR){for(let k=i;k<BR.length;k+=4)BR[k]=0}}
document.querySelectorAll('[data-mt]').forEach(b=>b.onclick=()=>{const t=+b.dataset.mt,i=MASKS.findIndex(m=>!m.t);if(!hasImg)return;if(i<0){toast('4 masks max. Delete one first');return}
 Object.assign(MASKS[i],{t,a:[.5,.5,.5,.5],f:t==2?.4:.4,inv:0,ev:0,con:0,sat:0,tmp:0});clearCh(i);mAct=i;mDraw={i,down:0};syncMask();toast(['','Drag across the photo','Drag outward from the centre','Tap a brightness','Tap a colour','Paint on the photo'][t])});
function pickMask(e,m){const r=cv.getBoundingClientRect(),x=Math.floor((e.clientX-r.left)/r.width*W),y=Math.floor((e.clientY-r.top)/r.height*H),p=new Uint8Array(4);gl.readPixels(x,H-1-y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,p);
 if(m.t==3){const L=(.2126*p[0]+.7152*p[1]+.0722*p[2])/255;m.a=[Math.max(0,L-.18),0,Math.min(1,L+.18),0];m.f=.12}
 else{const mx=Math.max(p[0],p[1],p[2]),mn=Math.min(p[0],p[1],p[2]),d=mx-mn;let hu=0;if(d)hu=mx==p[0]?((p[1]-p[2])/d+6)%6:mx==p[1]?(p[2]-p[0])/d+2:(p[0]-p[1])/d+4;m.a=[hu/6,0,.06,0];m.f=.08}}
function paintAt(e,slot){const r=cv.getBoundingClientRect(),bw=512,bh=Math.max(1,Math.round(512*H/W));if(!BR)BR=new Uint8Array(bw*bh*4);
 const cx=(e.clientX-r.left)/r.width*bw,cy=(e.clientY-r.top)/r.height*bh,rad=34*bw/r.width;
 for(let y=Math.max(0,Math.floor(cy-rad));y<Math.min(bh,cy+rad);y++)for(let x=Math.max(0,Math.floor(cx-rad));x<Math.min(bw,cx+rad);x++){const d=Math.hypot(x-cx,y-cy)/rad;if(d>=1)continue;const a=Math.round(255*(1-d*d)),k=((bh-1-y)*bw+x)*4+slot;if(a>BR[k])BR[k]=a}
 gl.activeTexture(gl.TEXTURE8);gl.bindTexture(gl.TEXTURE_2D,BRT);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,bw,bh,0,gl.RGBA,gl.UNSIGNED_BYTE,BR)}
function finishDraw(){mDraw=null;setTool('masks');syncMask();snap();draw()}
cv.addEventListener('pointerdown',e=>{if(!mDraw)return;e.stopImmediatePropagation();cv.setPointerCapture(e.pointerId);const m=MASKS[mDraw.i],[x,y]=mpos(e);mDraw.down=1;
 if(m.t==1)m.a=[x,y,x,y];else if(m.t==2)m.a=[x,y,.01,.01];else if(m.t==3||m.t==4){pickMask(e,m);finishDraw();return}else paintAt(e,mDraw.i);draw()},true);
cv.addEventListener('pointermove',e=>{if(!mDraw||!mDraw.down)return;e.stopImmediatePropagation();const m=MASKS[mDraw.i],[x,y]=mpos(e);
 if(m.t==1){m.a[2]=x;m.a[3]=y}else if(m.t==2){const rx=Math.max(.01,Math.hypot(x-m.a[0],(y-m.a[1])*H/W));m.a[2]=rx;m.a[3]=rx*W/H}else paintAt(e,mDraw.i);draw()},true);
cv.addEventListener('pointerup',e=>{if(!mDraw||!mDraw.down)return;e.stopImmediatePropagation();finishDraw()},true);
$('#minv').onclick=()=>{if(mAct>=0&&MASKS[mAct].t){MASKS[mAct].inv^=1;draw();snap()}};
$('#mshow').onclick=()=>{S.mShow=(S.mShow??-1)===mAct?-1:mAct;draw()};
$('#mdel').onclick=()=>{if(mAct>=0){MASKS[mAct].t=0;clearCh(mAct);if(BR){gl.activeTexture(gl.TEXTURE8);gl.bindTexture(gl.TEXTURE_2D,BRT);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,Math.round(512),Math.max(1,Math.round(512*H/W)),0,gl.RGBA,gl.UNSIGNED_BYTE,BR)}S.mShow=-1;mAct=MASKS.findIndex(m=>m.t);syncMask();draw();snap()}};
function guideUpdate(){if(!mkReady)return;const m=MASKS[mAct];if(!hasImg||!m||!m.t||(document.body.dataset.tool!=='masks'&&!mDraw)){mg.style.display='none';return}
 const r=cv.getBoundingClientRect();mg.width=r.width;mg.height=r.height;Object.assign(mg.style,{display:'block',left:r.left+'px',top:r.top+'px'});const g=mg.getContext('2d'),X=u=>u*r.width,Y=v=>(1-v)*r.height;g.strokeStyle='#fff';g.fillStyle='#fff';g.lineWidth=2;
 if(m.t==1){const[a,b,c,d]=m.a;g.beginPath();g.moveTo(X(a),Y(b));g.lineTo(X(c),Y(d));g.stroke();[[a,b],[c,d]].forEach(([x,y])=>{g.beginPath();g.arc(X(x),Y(y),7,0,7);g.fill()});
  const L=Math.hypot((c-a)*r.width,(d-b)*r.height)||1,nx=((d-b)*r.height)/L,ny=((c-a)*r.width)/L;g.setLineDash([6,6]);[[a,b],[c,d]].forEach(([x,y])=>{g.beginPath();g.moveTo(X(x)-nx*r.width,Y(y)-ny*r.width);g.lineTo(X(x)+nx*r.width,Y(y)+ny*r.width);g.stroke()})}
 else if(m.t==2){g.beginPath();g.ellipse(X(m.a[0]),Y(m.a[1]),m.a[2]*r.width,m.a[3]*r.height,0,0,7);g.stroke();g.setLineDash([5,6]);g.beginPath();g.ellipse(X(m.a[0]),Y(m.a[1]),m.a[2]*r.width*(1-m.f),m.a[3]*r.height*(1-m.f),0,0,7);g.stroke()}}
mkReady=1;syncMask();

// ================= GRADING WHEELS: Lift / Gamma / Gain (colour puck + level) =================
const WS={lift:{x:0,y:0,l:0},gam:{x:0,y:0,l:0},gain:{x:0,y:0,l:0}};
function wheelVec(w){const s=WS[w],r=Math.min(1,Math.hypot(s.x,s.y)),h=((Math.atan2(s.y,s.x)/6.2832)+1)%1,f=n=>{const k=(n+h*6)%6;return 1-Math.max(0,Math.min(k,4-k,1))},rgb=[f(5),f(3),f(1)],av=(rgb[0]+rgb[1]+rgb[2])/3;return rgb.map(v=>(v-av)*r*1.5+s.l)}
function drawWheel(w){const c=document.querySelector('.wheel[data-w="'+w+'"] canvas');if(!c)return;const g=c.getContext('2d'),R=62,C=70;g.clearRect(0,0,140,140);
 const cg=g.createConicGradient?g.createConicGradient(0,C,C):null;if(cg){for(let i=0;i<=12;i++)cg.addColorStop(i/12,'hsl('+(i*30)+',85%,55%)');g.fillStyle=cg}else g.fillStyle='#777';
 g.beginPath();g.arc(C,C,R,0,7);g.fill();const rg=g.createRadialGradient(C,C,0,C,C,R);rg.addColorStop(0,'rgba(24,24,26,.95)');rg.addColorStop(.7,'rgba(24,24,26,.15)');rg.addColorStop(1,'rgba(24,24,26,0)');g.fillStyle=rg;g.beginPath();g.arc(C,C,R,0,7);g.fill();
 g.strokeStyle='rgba(255,255,255,.35)';g.lineWidth=1.5;g.beginPath();g.arc(C,C,R,0,7);g.stroke();const s=WS[w];g.fillStyle='#fff';g.beginPath();g.arc(C+s.x*R,C+s.y*R,8,0,7);g.fill();g.strokeStyle='#000';g.lineWidth=2;g.stroke()}
const drawWheels=()=>['lift','gam','gain'].forEach(drawWheel);
function updW(w){WH[w]=wheelVec(w);drawWheel(w);draw()}
document.querySelectorAll('.wheel').forEach(el=>{const w=el.dataset.w,c=el.querySelector('canvas');
 const mv=e=>{const r=c.getBoundingClientRect();let x=(e.clientX-r.left)/r.width*2-1,y=(e.clientY-r.top)/r.height*2-1;const m=Math.hypot(x,y);if(m>1){x/=m;y/=m}WS[w].x=x;WS[w].y=y;updW(w)};
 c.onpointerdown=e=>{c.setPointerCapture(e.pointerId);mv(e);c.onpointermove=mv};c.onpointerup=()=>{c.onpointermove=null;snap()};c.ondblclick=()=>{Object.assign(WS[w],{x:0,y:0});updW(w);snap()}});
[['wlLift','lift'],['wlGam','gam'],['wlGain','gain']].forEach(([id,w])=>$('#'+id).addEventListener('input',e=>{WS[w].l=+e.target.value;updW(w)}));
$('#reset').addEventListener('click',()=>{for(const w in WS)Object.assign(WS[w],{x:0,y:0,l:0});['wlLift','wlGam','wlGain'].forEach(id=>$('#'+id).value=0);drawWheels()});
drawWheels();
// ================= GOBO light patterns =================
document.querySelectorAll('[data-g]').forEach(b=>b.onclick=()=>{S.gType=+b.dataset.g;if(S.gType&&!S.gAmt){S.gAmt=ins.gAmt.value=.6;sync('gAmt')}draw();snap()});
// ================= GRADE RAMP video (start look -> end look) =================
let RA=null,RB=null;const snapS=()=>{const o={};for(const k in DEF)o[k]=S[k]||0;return o};
$('#rampA').onclick=()=>{RA=snapS();toast('Start look set')};$('#rampB').onclick=()=>{RB=snapS();toast('End look set')};
$('#rampGo').onclick=()=>{if(!hasImg)return;if(!RA||!RB){toast('Set a start and an end look first');return}if(!window.MediaRecorder){toast('Recording is not supported here');return}
 const keep={...S},mt=MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm',rec=new MediaRecorder(cv.captureStream(30),{mimeType:mt}),ch=[];rec.ondataavailable=e=>ch.push(e.data);
 rec.onstop=()=>{Object.assign(S,keep);S.anim=keep.anim;for(const k in DEF){ins[k].value=S[k];sync(k)}draw();dl(new Blob(ch,{type:'video/webm'}),fname('_ramp.webm'))};
 const d=6000,t0=performance.now();S.anim=true;rec.start();toast('Recording 6 seconds');
 (function f(now){const p=Math.min(1,(now-t0)/d),e=p<.5?2*p*p:1-Math.pow(-2*p+2,2)/2;for(const k in DEF)S[k]=RA[k]+(RB[k]-RA[k])*e;draw();if(p<1)requestAnimationFrame(f);else rec.stop()})(t0)};
// ================= MASK RAIL (left edge): tap a notch to jump between masks =================
var mr=document.createElement('div');mr.id='mrail';document.body.append(mr);
function renderRail(){if(!mr)return;mr.innerHTML='';MASKS.forEach((m,i)=>{if(!m.t)return;const b=document.createElement('button');b.textContent=i+1;b.className=i===mAct?'on':'';b.title=MN[m.t];b.onclick=()=>{mAct=i;syncMask();setTool('masks');draw()};mr.append(b)});mr.style.display=mr.children.length?'flex':'none'}
renderRail();

// ================= HOME PAGE AESTHETICS: mood dial, viewfinder, engraved lens =================
const MOODS=[
['noir',{g1:'#2a2a2e',g2:'#000',rf:'grayscale(1) contrast(1.2) brightness(.32)',rfl:'grayscale(1) contrast(1.25)',cone:'255,255,255',stroke:'rgba(255,255,255,.16)',read:['f/2.0','1/48','ISO 3200','3200K'],look:['Black & White','Ilford HP5']}],
['golden',{g1:'#ff9a3c',g2:'#3a1d4a',rf:'sepia(.6) saturate(1.2) brightness(.34)',rfl:'sepia(.35) saturate(1.2)',cone:'255,210,140',stroke:'rgba(255,190,100,.24)',read:['f/2.8','1/48','ISO 200','4300K'],look:['Arthouse','Iranian Golden Hour']}],
['neon',{g1:'#ff2bd6',g2:'#071a3a',rf:'saturate(1.5) hue-rotate(-15deg) brightness(.34)',rfl:'saturate(1.5) hue-rotate(-15deg)',cone:'120,220,255',stroke:'rgba(80,220,255,.26)',read:['f/1.4','1/48','ISO 1600','4000K'],look:['Arthouse','Tokyo Neon']}],
['darkroom',{g1:'#6b0a0a',g2:'#150000',rf:'sepia(1) hue-rotate(-40deg) saturate(2) brightness(.3)',rfl:'sepia(1) hue-rotate(-40deg) saturate(2) brightness(.8)',cone:'255,60,40',stroke:'rgba(255,70,50,.26)',read:['f/8','1/15','ISO 100','SAFELIGHT'],look:['Black & White','Tri-X 400 +1']}],
['silent',{g1:'#bba98a',g2:'#1b1710',rf:'sepia(.9) contrast(1.1) brightness(.33)',rfl:'sepia(.8) contrast(1.1)',cone:'255,240,200',stroke:'rgba(240,225,190,.22)',read:['f/4','1/40','ISO 50','ORTHO'],look:['Black & White','Ortho Cinema']}],
['technicolor',{g1:'#e8432b',g2:'#0e3a4a',rf:'saturate(1.9) brightness(.36)',rfl:'saturate(1.9)',cone:'255,230,120',stroke:'rgba(255,120,80,.26)',read:['f/5.6','1/48','ISO 50','5600K'],look:['Vintage','Technicolor 3-Strip']}],
['dream',{g1:'#ffb7d5',g2:'#2a1a4a',rf:'saturate(.9) brightness(.38) hue-rotate(-10deg)',rfl:'saturate(.9) brightness(1.05)',cone:'255,190,230',stroke:'rgba(255,190,230,.26)',read:['f/1.8','1/60','ISO 100','5200K'],look:['Arthouse','French New Wave']}],
['desert',{g1:'#e07a3a',g2:'#1d3a46',rf:'sepia(.5) saturate(1.3) brightness(.34)',rfl:'sepia(.3) saturate(1.3)',cone:'255,190,120',stroke:'rgba(255,170,100,.26)',read:['f/8','1/250','ISO 100','6000K'],look:['Genre','Neo-Western Amber']}],
['arctic',{g1:'#9fd2ff',g2:'#0a1c33',rf:'saturate(.7) hue-rotate(10deg) brightness(.36)',rfl:'saturate(.7) hue-rotate(10deg)',cone:'190,230,255',stroke:'rgba(170,220,255,.28)',read:['f/5.6','1/500','ISO 100','7500K'],look:['Arthouse','Nordic Cold']}],
['forest',{g1:'#2f7a4a',g2:'#04120a',rf:'hue-rotate(60deg) saturate(1.3) brightness(.32)',rfl:'hue-rotate(60deg) saturate(1.2)',cone:'170,255,190',stroke:'rgba(140,240,170,.26)',read:['f/4','1/125','ISO 800','5000K'],look:['Genre','Thriller Emerald']}],
['rain',{g1:'#5b7fa0',g2:'#0b1118',rf:'saturate(.6) hue-rotate(-10deg) brightness(.33)',rfl:'saturate(.7) hue-rotate(-10deg)',cone:'180,210,240',stroke:'rgba(160,200,235,.24)',read:['f/2.0','1/50','ISO 1600','4200K'],look:['35mm Motion','Vision3 500T']}],
['seventies',{g1:'#c9a227',g2:'#2d1a0a',rf:'sepia(.8) saturate(1.1) brightness(.34)',rfl:'sepia(.55) saturate(1.1)',cone:'255,220,140',stroke:'rgba(230,200,110,.26)',read:['f/2.8','1/60','ISO 400','4800K'],look:['Vintage','Agfacolor Neu']}]];
let MOODLOOK=null,MOODNAME='',moodI=-1;
// engraved lens markings on both rings
const ENG=['LUMINA','35mm','T1.5','\u221e','3','1.5','1','0.7','\u00b7','SN 0042','LUMINA','50','25','T2.8','\u221e','10','5','3','1.5','1','\u00b7','CINE','24','\u00b7'];
function engrave(r){ENG.forEach((t,i)=>{const e=document.createElement('em');e.className='eng';e.textContent=t;e.style.transform='rotate('+i*15+'deg) translateY(calc(var(--R) * -1 + 15px)) translateX(-50%)';r.append(e)})}
engrave(barrel);
// the mood ring (home page only)
const moodR=document.createElement('div');moodR.id='mood';ui.append(moodR);
for(let i=0;i<12;i++){const e=document.createElement('em');e.textContent=MOODS[i][0];e.style.transform='rotate('+i*30+'deg) translateY(calc(var(--R) * -1 + 46px)) translateX(-50%)';moodR.append(e)}
engrave(moodR);
const moodLit=i=>[...moodR.children].forEach((e,j)=>e.classList.toggle('on',j<12&&j===i));
function setMood(i,auto){const k=i%MOODS.length,m=MOODS[k][1],s=document.body.style;document.body.dataset.mood=MOODS[k][0];
 s.setProperty('--g1',m.g1);s.setProperty('--g2',m.g2);s.setProperty('--rf',m.rf);s.setProperty('--rfl',m.rfl);s.setProperty('--cone',m.cone);s.setProperty('--stroke',m.stroke);
 setRead(m.read);MOODLOOK=auto?null:m.look;MOODNAME=MOODS[k][0];const g=$('#greet');if(g)g.textContent='mood \u00b7 '+MOODS[k][0];if(!auto){burn();clk()}}
function moodLook(){if(MOODLOOK&&!cp){const[d,n]=MOODLOOK;deck=d;cp=DECKS[d][n];renderDeck();applyCP();slate('Mood: '+MOODNAME)}MOODLOOK=null}
let mdr=null,mrot=0;
moodR.onpointerdown=e=>{moodR.setPointerCapture(e.pointerId);const b=moodR.getBoundingClientRect(),cx=b.left+b.width/2,cy=b.top+b.height/2;mdr={cx,cy,a:Math.atan2(e.clientY-cy,e.clientX-cx),r0:mrot}};
moodR.onpointermove=e=>{if(!mdr)return;let d=(Math.atan2(e.clientY-mdr.cy,e.clientX-mdr.cx)-mdr.a)*57.2958;d=((d+540)%360)-180;mrot=mdr.r0+d;moodR.style.transition='none';moodR.style.transform='rotate('+mrot+'deg)';
 const i=((Math.round(-mrot/30)%12)+12)%12;if(i!==moodI){moodI=i;moodLit(i);navigator.vibrate&&navigator.vibrate(4)}};
moodR.onpointerup=()=>{if(!mdr)return;mdr=null;mrot=Math.round(mrot/30)*30;moodR.style.transition='transform .6s cubic-bezier(.2,1.25,.4,1)';moodR.style.transform='rotate('+mrot+'deg)';const i=((Math.round(-mrot/30)%12)+12)%12;moodI=i;moodLit(i);setMood(i)};
// viewfinder overlay: brackets, REC, running timecode, exposure readout
const vf=document.createElement('div');vf.id='vf';vf.innerHTML='<i class="c tl"></i><i class="c tr"></i><i class="c bl"></i><i class="c br"></i><div id="vfl"><u></u>REC <span id="vtc">00:00:00:00</span></div><div id="vfr"></div>';document.body.append(vf);
function setRead(r){$('#vfr').textContent=r[0]+' \u00b7 '+r[1]+' \u00b7 '+r[2]+' \u00b7 '+r[3]}setRead(['f/1.8','1/48','ISO 400','5600K']);
setInterval(()=>{if(document.hidden)return;const t=performance.now()/1000;$('#vtc').textContent=[Math.floor(t/3600),Math.floor(t/60)%60,Math.floor(t)%60,Math.floor(t*24)%24].map(v=>String(v).padStart(2,'0')).join(':')},83);

// ================= HOME PAGE: rack focus, film feel, flare, breathing, letterbox, attract mode, dust =================
const mk=(tag,id,html)=>{const e=document.createElement(tag);e.id=id;if(html)e.innerHTML=html;document.body.append(e);return e};
mk('div','grain');mk('div','flare','<i></i><i></i>');mk('div','lb1').className='lbar';mk('div','lb2').className='lbar';mk('div','streakB');const motes=mk('canvas','motes');
moodR.addEventListener('pointermove',()=>{if(!mdr)return;const a=Math.abs(Math.sin(mrot*.0524))*5,s=document.body.style,em=$('#empty');s.setProperty('--rfa',a+'px');s.setProperty('--rfb',(5-a)+'px');if(em)em.style.transform='scale(1.012)'});
moodR.addEventListener('pointerup',()=>{const s=document.body.style,em=$('#empty');s.setProperty('--rfa','0px');s.setProperty('--rfb','0px');if(em)em.style.transform='';document.body.classList.add('lbox')});
['open','sample','liveBtn'].forEach(id=>{const b=$('#'+id);if(b)b.addEventListener('pointerdown',()=>{const s=$('#streakB');s.classList.remove('go');void s.offsetWidth;s.classList.add('go')})});
(function(){const g=motes.getContext('2d'),P=Array.from({length:46},()=>({x:Math.random(),y:Math.random(),vx:(Math.random()-.5)*.00012,vy:-Math.random()*.00018-.00004,r:Math.random()*1.6+.5}));
 const rs=()=>{motes.width=innerWidth;motes.height=innerHeight};rs();addEventListener('resize',rs);
 (function f(){requestAnimationFrame(f);if(document.body.classList.contains('has')||document.hidden)return;g.clearRect(0,0,motes.width,motes.height);
  const bx=parseFloat(document.body.style.getPropertyValue('--bx'))||innerWidth/2,by=parseFloat(document.body.style.getPropertyValue('--by'))||innerHeight/2;
  P.forEach(p=>{p.x+=p.vx*16;p.y+=p.vy*16;if(p.y<-.02){p.y=1.02;p.x=Math.random()}if(p.x<0||p.x>1)p.vx*=-1;const X=p.x*motes.width,Y=p.y*motes.height,d=Math.hypot(X-bx,Y-by),a=.12+.88*Math.max(0,1-d/230);g.fillStyle='rgba(255,236,200,'+a.toFixed(2)+')';g.beginPath();g.arc(X,Y,p.r,0,7);g.fill()});
  if(Math.random()<.03){g.strokeStyle='rgba(255,255,255,.25)';g.beginPath();const x=Math.random()*motes.width;g.moveTo(x,0);g.lineTo(x+Math.random()*6,motes.height);g.stroke()}})()})();
let idle=Date.now(),att=0;const rate=r=>document.getAnimations().forEach(a=>{if(a.animationName==='drift')a.playbackRate=r});
['pointerdown','pointermove','touchstart','keydown'].forEach(ev=>addEventListener(ev,()=>{idle=Date.now();if(att){att=0;rate(1)}},{passive:true}));
setInterval(()=>{const f=$('#fun');if(document.body.classList.contains('has')||(f&&f.classList.contains('on'))||document.hidden)return;
 if(Date.now()-idle>20000){if(!att){att=1;rate(5)}mrot-=30;moodR.style.transition='transform .9s cubic-bezier(.2,1.25,.4,1)';moodR.style.transform='rotate('+mrot+'deg)';const i=((Math.round(-mrot/30)%12)+12)%12;moodI=i;moodLit(i);setMood(i,true)}},4000);
// ================= FUN ZONE =================
function sunTimes(lat,lng,date){const R=Math.PI/180,D=864e5,toJ=d=>d/D-.5+2440588,fromJ=j=>new Date((j+.5-2440588)*D),J0=2451545,lw=R*-lng,phi=R*lat,e=R*23.4397,d=toJ(date)-J0,n=Math.round(d-.0009-lw/(2*Math.PI)),ds=.0009+lw/(2*Math.PI)+n,M=R*(357.5291+.98560028*ds),C=R*(1.9148*Math.sin(M)+.02*Math.sin(2*M)+.0003*Math.sin(3*M)),L=M+C+R*102.9372+Math.PI,dec=Math.asin(Math.sin(e)*Math.sin(L)),Jt=J0+ds+.0053*Math.sin(M)-.0069*Math.sin(2*L);
 const pair=h=>{const w=Math.acos((Math.sin(h*R)-Math.sin(phi)*Math.sin(dec))/(Math.cos(phi)*Math.cos(dec))),Js=J0+(.0009+(w+lw)/(2*Math.PI)+n)+.0053*Math.sin(M)-.0069*Math.sin(2*L);return[fromJ(Jt-(Js-Jt)),fromJ(Js)]};
 const a=pair(-.833),b=pair(6),c=pair(-8);return{rise:a[0],set:a[1],gAM:b[0],gPM:b[1],bAM:c[0],bPM:c[1]}}
const CAMS=[['Sony A7 IV','Full-frame',33,'4K60',5.5,658,2498],['Sony A7C II','Full-frame',33,'4K60',7,514,2199],['Sony A6700','APS-C',26,'4K120',5,493,1398],['Sony ZV-E10 II','APS-C',26,'4K60',0,377,999],['Canon EOS R6 II','Full-frame',24,'4K60',8,670,2499],['Canon EOS R50','APS-C',24,'4K30',0,375,679],['Nikon Z6 III','Full-frame',24,'6K60 RAW',8,760,2499],['Nikon Zf','Full-frame',24,'4K30',8,710,1999],['Fujifilm X-T5','APS-C',40,'6.2K30',7,557,1699],['Fujifilm X100VI','APS-C',40,'6.2K30',6,521,1599],['Panasonic S5 II','Full-frame',24,'6K30',6.5,740,1999],['OM System OM-1 II','Micro 4/3',20,'4K60',8.5,599,2400],['Galaxy S24 Ultra','Phone',200,'8K30',0,232,1299],['iPhone 16 Pro Max','Phone',48,'4K120',0,227,1199],['Pixel 9 Pro XL','Phone',50,'4K60',0,221,1099]];
function cmpCam(el){const FX={US:['$',1,'com'],India:['\u20b9',88,'co.in'],UK:['\u00a3',.78,'co.uk'],Europe:['\u20ac',.92,'de']},o=CAMS.map((c,i)=>'<option value="'+i+'">'+c[0]+'</option>').join('');
 el.innerHTML='<div class="r2"><select id="ca">'+o+'</select><span>vs</span><select id="cb">'+o+'</select></div><div class="r2"><select id="cr">'+Object.keys(FX).map(k=>'<option>'+k+'</option>').join('')+'</select></div><div id="ct"></div><p class="note">Specs and prices are US launch figures from memory, converted at rough rates. They can be wrong or out of date. Tap "live price" to check the real shop price in your region.</p>';
 el.querySelector('#cb').value=1;
 const go=()=>{const A=CAMS[el.querySelector('#ca').value],B=CAMS[el.querySelector('#cb').value],rg=el.querySelector('#cr').value,f=FX[rg],pr=c=>f[0]+Math.round(c[6]*f[1]).toLocaleString(),st=c=>c[4]?c[4]+' stops':'none / lens-based',
  row=(l,a,b,w)=>'<tr><th>'+l+'</th><td class="'+(w==1?'win':'')+'">'+a+'</td><td class="'+(w==2?'win':'')+'">'+b+'</td></tr>',cp=(x,y,hi)=>x===y?0:((x>y)===hi?1:2),v=[];
  if(A[2]!==B[2])v.push((A[2]>B[2]?A[0]:B[0])+' has more megapixels');if(A[5]!==B[5])v.push((A[5]<B[5]?A[0]:B[0])+' is lighter to carry');if(A[6]!==B[6])v.push((A[6]<B[6]?A[0]:B[0])+' costs less');if(A[4]!==B[4])v.push((A[4]>B[4]?A[0]:B[0])+' has stronger stabilisation');
  const lk=c=>'<a target="_blank" rel="noopener" href="https://www.google.'+f[2]+'/search?tbm=shop&q='+encodeURIComponent(c[0])+'">live price</a>';
  el.querySelector('#ct').innerHTML='<table><tr><th></th><th>'+A[0]+'</th><th>'+B[0]+'</th></tr>'+row('Sensor',A[1],B[1],0)+row('Megapixels',A[2],B[2],cp(A[2],B[2],true))+row('Video',A[3],B[3],0)+row('Stabilisation',st(A),st(B),cp(A[4],B[4],true))+row('Weight',A[5]+' g',B[5]+' g',cp(A[5],B[5],false))+row('Launch price ('+rg+')',pr(A),pr(B),cp(A[6],B[6],false))+row('Check',lk(A),lk(B),0)+'</table><p class="verdict">'+v.join('. ')+'.</p>'};
 ['#ca','#cb','#cr'].forEach(s=>el.querySelector(s).onchange=go);go()}
function posterGame(el){const G={Thriller:{f:'grayscale(.7) contrast(1.3) brightness(.8)',a:['THE SILENT','DEAD OF','NO WAY','BEHIND THE'],b:['WITNESS','NIGHT','OUT','DOOR'],g:'Trust no one.',c:'#e8e8e8'},Romance:{f:'sepia(.35) saturate(1.2) brightness(1.05)',a:['LAST','LOVE IN','A LETTER','SUMMER'],b:['SUMMER','PARIS','TO YOU','RAIN'],g:'Some stories find you.',c:'#ffd9c9'},'Sci-Fi':{f:'hue-rotate(160deg) saturate(1.3) contrast(1.15)',a:['BEYOND','PROJECT','THE LAST','ORBIT'],b:['ANDROMEDA','NOVA','SIGNAL','ZERO'],g:'We are not alone.',c:'#aff'},Western:{f:'sepia(.7) contrast(1.2)',a:['HIGH','THE LONG','DUST AND','TWO'],b:['NOON','ROAD','IRON','REVOLVERS'],g:'Justice has a price.',c:'#f3d9a8'},Horror:{f:'grayscale(.5) contrast(1.4) brightness(.7) hue-rotate(-20deg)',a:['DONT LOOK','THE HOUSE','SLEEP','THE THING'],b:['BEHIND','AT NIGHT','NEVER','BELOW'],g:'It is already inside.',c:'#ff9a9a'},Comedy:{f:'saturate(1.5) brightness(1.1)',a:['MY BIG','THE WORST','OOPS','DINNER'],b:['MISTAKE','BEST MAN','WRONG DAY','DISASTER'],g:'What could go wrong?',c:'#fff3a8'}};
 el.innerHTML='<canvas id="pc" width="600" height="900"></canvas><div id="pg">'+Object.keys(G).map(k=>'<button class="btn" data-k="'+k+'">'+k+'</button>').join('')+'</div><div><button class="btn" id="pr">New title</button><label class="btn">Use my photo<input id="pf" type="file" accept="image/*" hidden></label><button class="btn" id="ps">Save poster</button></div>';
 const c=el.querySelector('#pc'),g=c.getContext('2d');let img=null,k='Thriller',t=[0,0];const rnd=()=>{const q=G[k];t=[Math.floor(Math.random()*q.a.length),Math.floor(Math.random()*q.b.length)]};
 const paint=()=>{const q=G[k],s=img||sampleCanvas(),sc=Math.max(600/s.width,900/s.height);g.clearRect(0,0,600,900);g.filter=q.f;g.drawImage(s,(600-s.width*sc)/2,(900-s.height*sc)/2,s.width*sc,s.height*sc);g.filter='none';
  let gr=g.createLinearGradient(0,380,0,900);gr.addColorStop(0,'rgba(0,0,0,0)');gr.addColorStop(1,'rgba(0,0,0,.85)');g.fillStyle=gr;g.fillRect(0,380,600,520);gr=g.createRadialGradient(300,450,250,300,450,650);gr.addColorStop(0,'rgba(0,0,0,0)');gr.addColorStop(1,'rgba(0,0,0,.55)');g.fillStyle=gr;g.fillRect(0,0,600,900);
  g.textAlign='center';g.fillStyle=q.c;g.font='700 34px JetBrains Mono,monospace';g.fillText(q.a[t[0]],300,690);let fs=96,tx=q.b[t[1]];g.font='700 '+fs+'px Bricolage Grotesque,system-ui,sans-serif';while(g.measureText(tx).width>540&&fs>40){fs-=4;g.font='700 '+fs+'px Bricolage Grotesque,system-ui,sans-serif'}g.fillText(tx,300,780);
  g.font='italic 22px system-ui';g.fillText(q.g,300,818);g.font='500 13px JetBrains Mono,monospace';g.fillStyle='#bbb';g.fillText('A LUMINA STUDIO PRODUCTION  \u00b7  COMING SOON',300,868)};
 el.querySelectorAll('[data-k]').forEach(b=>b.onclick=()=>{k=b.dataset.k;rnd();paint()});el.querySelector('#pr').onclick=()=>{rnd();paint()};
 el.querySelector('#pf').onchange=async e=>{if(e.target.files[0]){img=await createImageBitmap(e.target.files[0]);paint()}};el.querySelector('#ps').onclick=()=>c.toBlob(b=>dl(b,'poster_'+k+'.png'));paint()}
function guessGame(el){const items=(window.HERO||[]).filter(x=>!x[0].includes('\u00b7'));if(items.length<4){el.innerHTML='<p>The looks are still loading. Close and try again in a moment.</p>';return}
 let score=0,round=0;const next=()=>{if(round==8){el.innerHTML='<h2>'+score+' / 8</h2><p>'+(score>=7?'Cinematographer':score>=5?'Film buff':score>=3?'Casual viewer':'Fresh to film')+'</p><button class="btn" id="ag">Play again</button>';el.querySelector('#ag').onclick=()=>{score=0;round=0;next()};return}
  const it=items[Math.floor(Math.random()*items.length)],o=[it];while(o.length<4){const x=items[Math.floor(Math.random()*items.length)];if(!o.some(y=>y[0]===x[0]))o.push(x)}o.sort(()=>Math.random()-.5);
  el.innerHTML='<p>Round '+(round+1)+' of 8 \u00b7 score '+score+'</p><h2>Which picture is \u201c'+it[0]+'\u201d?</h2><div class="gg">'+o.map((x,i)=>'<button data-i="'+i+'"><img src="'+x[2]+'" alt=""></button>').join('')+'</div>';
  el.querySelectorAll('.gg button').forEach(b=>b.onclick=()=>{const ok=o[+b.dataset.i]===it;if(ok)score++;el.querySelectorAll('.gg button').forEach(x=>{x.disabled=true;x.style.outline=o[+x.dataset.i]===it?'3px solid #2ecc71':'none'});if(!ok)b.style.outline='3px solid #ff3b30';navigator.vibrate&&navigator.vibrate(ok?10:[30,40,30]);round++;setTimeout(next,1000)})};next()}
function expoToy(el){el.innerHTML='<canvas id="xc" width="320" height="200"></canvas><div id="xm"></div><label>Aperture <b id="av"></b><input id="ap" type="range" min="0" max="7" value="6"></label><label>Shutter <b id="sv"></b><input id="sh2" type="range" min="0" max="8" value="3"></label><label>ISO <b id="iv"></b><input id="is" type="range" min="0" max="6" value="0"></label><p class="note">Small f-number blurs the background. Slow shutter smears movement. High ISO adds grain and brightness.</p>';
 const AP=[1.4,2,2.8,4,5.6,8,11,16],SH=[1000,500,250,125,60,30,15,8,4],IS=[100,200,400,800,1600,3200,6400],c=el.querySelector('#xc'),g=c.getContext('2d'),q=s=>el.querySelector(s);let raf;const t0=performance.now();
 const f=now=>{raf=requestAnimationFrame(f);const N=AP[+q('#ap').value],T=1/SH[+q('#sh2').value],I=IS[+q('#is').value],ex=(T/(1/125))*(I/100)/(N*N/121),bl=Math.max(0,(11-N)*.7),x=((now-t0)/1000*400)%(c.width+80)-40;
  q('#av').textContent='f/'+N;q('#sv').textContent=SH[+q('#sh2').value]>=4?'1/'+SH[+q('#sh2').value]:'';q('#iv').textContent=I;
  g.filter='blur('+bl+'px)';let gr=g.createLinearGradient(0,0,0,200);gr.addColorStop(0,'#6ab7ff');gr.addColorStop(.6,'#ffd9a0');gr.addColorStop(.6,'#2d6a3a');gr.addColorStop(1,'#163a20');g.fillStyle=gr;g.fillRect(-10,-10,340,220);g.fillStyle='#fff';for(let i=0;i<6;i++)g.fillRect(20+i*55,60+(i%2)*14,14,70);g.filter='none';
  const len=400*T,n=Math.max(1,Math.min(14,Math.round(len/3)));for(let k=n;k>=0;k--){g.globalAlpha=k?.5/n:1;g.fillStyle='#ff4d4d';g.beginPath();g.arc(x-k*len/n,140,16,0,7);g.fill()}g.globalAlpha=1;
  const dots=I/100*30;for(let i=0;i<dots;i++){g.fillStyle=Math.random()<.5?'rgba(255,255,255,.5)':'rgba(0,0,0,.5)';g.fillRect(Math.random()*320,Math.random()*200,1.5,1.5)}
  c.style.filter='brightness('+Math.min(3,Math.max(.12,ex))+')';q('#xm').textContent=ex<.5?'Too dark':ex>2?'Too bright':'Well exposed'};raf=requestAnimationFrame(f);return()=>cancelAnimationFrame(raf)}
function goldenPlan(el){el.innerHTML='<button class="btn" id="loc">Use my location</button><div id="gp"></div><p class="note">Calculated on your phone from your position and today\u2019s date. Nothing is sent anywhere.</p>';
 const out=el.querySelector('#gp'),fm=t=>t&&!isNaN(t)?t.toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'}):'-',show=(la,lo)=>{const d=new Date(),T=sunTimes(la,lo,d),m0=new Date(d).setHours(0,0,0,0),fr=t=>t&&!isNaN(t)?Math.max(0,Math.min(100,(t-m0)/864e5*100)).toFixed(1):0,
  stops=[[0,'#0a1030'],[fr(T.bAM),'#2a4cff'],[fr(T.rise),'#ff9a3c'],[fr(T.gAM),'#9ec8ff'],[fr(T.gPM),'#ff9a3c'],[fr(T.set),'#2a4cff'],[fr(T.bPM),'#0a1030'],[100,'#0a1030']];
  out.innerHTML='<div id="tl" style="background:linear-gradient(90deg,'+stops.map(s=>s[1]+' '+s[0]+'%').join(',')+')"><i style="left:'+fr(d)+'%"></i></div><table><tr><th>Blue hour starts</th><td>'+fm(T.bAM)+'</td></tr><tr><th>Sunrise</th><td>'+fm(T.rise)+'</td></tr><tr><th>Morning golden hour ends</th><td>'+fm(T.gAM)+'</td></tr><tr><th>Evening golden hour starts</th><td>'+fm(T.gPM)+'</td></tr><tr><th>Sunset</th><td>'+fm(T.set)+'</td></tr><tr><th>Blue hour ends</th><td>'+fm(T.bPM)+'</td></tr></table><p class="verdict">Best light for photos: '+fm(T.gPM)+' to '+fm(T.set)+' this evening.</p>'};
 el.querySelector('#loc').onclick=()=>navigator.geolocation?navigator.geolocation.getCurrentPosition(p=>show(p.coords.latitude,p.coords.longitude),()=>{out.textContent='Location was blocked. Allow it in your browser settings to use this.'}):out.textContent='Location is not available here.'}
function quizGame(el){const P=[['Kodachrome 64','Warm, bold and nostalgic. You turn ordinary days into postcards.'],['Ilford HP5','Quiet and timeless. You notice what others walk past.'],['Tokyo Neon','Restless and electric. The night is your home.'],['Nordic Cold','Calm and crisp. You like space, silence and clean lines.'],['Technicolor 3-Strip','Joyful and larger than life. Every day is a musical.'],['Iranian Golden Hour','Warm and poetic. You live for the last light of the day.']],
 Q=[['Your ideal evening?',[['Rooftop dinner at sunset',5],['Neon city walk',2],['A book and silence',1],['Road trip, windows down',0]]],['Pick a weather',[['Crisp snowy morning',3],['Golden afternoon',5],['Thunderstorm',1],['Bright blue sky',4]]],['Your soundtrack?',[['Old classics',0],['Synthwave',2],['Soft piano',3],['Big orchestral musical',4]]],['Friends call you',[['The dreamer',5],['The storyteller',0],['The night owl',2],['The calm one',3]]],['Pick a decade',[['1920s',1],['1950s',4],['1970s',0],['2080s',2]]]];
 let qi=0,sc=[0,0,0,0,0,0];const next=()=>{if(qi==Q.length){const w=sc.indexOf(Math.max(...sc)),it=(window.HERO||[]).find(x=>x[0]===P[w][0]);el.innerHTML='<h2>You are '+P[w][0]+'</h2>'+(it?'<img class="gimg" src="'+it[2]+'">':'')+'<p>'+P[w][1]+'</p><button class="btn" id="qa">Again</button>';el.querySelector('#qa').onclick=()=>{qi=0;sc=[0,0,0,0,0,0];next()};return}
  el.innerHTML='<p>Question '+(qi+1)+' of '+Q.length+'</p><h2>'+Q[qi][0]+'</h2>'+Q[qi][1].map((o,i)=>'<button class="btn qo" data-i="'+i+'">'+o[0]+'</button>').join('');
  el.querySelectorAll('.qo').forEach(b=>b.onclick=()=>{sc[Q[qi][1][+b.dataset.i][1]]++;qi++;navigator.vibrate&&navigator.vibrate(8);next()})};next()}
const FUN=[['Camera Face-Off','Compare specs and prices',cmpCam],['Movie Poster Maker','Your photo, a genre, a title',posterGame],['Guess the Look','Name that film style',guessGame],['Exposure Playground','Play with aperture, shutter, ISO',expoToy],['Golden Hour Planner','When is the light best today?',goldenPlan],['Which Film Are You?','A five-question quiz',quizGame]];
const fun=mk('div','fun','<div class="curt l"></div><div class="curt r"></div><div id="funIn"><header><button id="funBack" hidden>\u2190 back</button><b id="funT">Fun Zone</b><button id="funX">close</button></header><p class="sub"></p><div id="stubs"></div><div id="tickets"></div><div id="funBody" hidden></div></div>');
let funClean=null;const tk=$('#tickets'),fb=$('#funBody');
FUN.push(...FUN2());FUN.forEach(([t,s,fn],i)=>{const b=document.createElement('button');b.className='tk';b.style.animationDelay=(.9+i*.09)+'s';b.innerHTML='<small>ADMIT ONE \u00b7 0'+(i+1)+'</small><b>'+t+'</b><span>'+s+'</span>';b.onclick=()=>{if(t!=='Roll the Credits'&&!tixSpend(b))return;tk.hidden=true;$('.sub',fun)&&($('.sub').hidden=true);$('#stubs').hidden=true;fb.hidden=false;$('#funBack').hidden=false;$('#funT').textContent=t;fb.innerHTML='';funClean=fn(fb)||null};tk.append(b)});
function funHome(){if(funClean){funClean();funClean=null}fb.hidden=true;fb.innerHTML='';tk.hidden=false;$('.sub').hidden=false;$('#stubs').hidden=false;$('#funBack').hidden=true;$('#funT').textContent='Fun Zone'}
$('#funBack').onclick=funHome;
function openFun(){funHome();fun.classList.add('on');void fun.offsetWidth;fun.classList.add('open');shatter()}
$('#funX').onclick=()=>{funHome();fun.classList.remove('open');setTimeout(()=>fun.classList.remove('on'),450)};
const funB=document.createElement('button');funB.className='rb';funB.title='Fun Zone';funB.innerHTML='<svg viewBox="0 0 24 24"><path d="M12 3l2.2 5.6L20 11l-5.8 2.4L12 19l-2.2-5.6L4 11l5.8-2.4z"/></svg>';camB.before(funB);funB.onclick=openFun;

// ================= BACK TO HOME, MY LOOKS, SHARE SIZES =================
const homeB=document.createElement('button');homeB.className='rb';homeB.id='homeB';homeB.title='Back to home';homeB.innerHTML='<svg viewBox="0 0 24 24"><path d="M15 5l-7 7 7 7"/></svg>';$('#top').prepend(homeB);
homeB.onclick=async()=>{if(!liveOn&&hasImg)await idb('put','state',{s:state(),t:thumbBig()});try{sessionStorage.setItem('skipIntro','1')}catch(e){}if(liveOn)live(false);location.reload()};
try{const m=JSON.parse(localStorage.getItem('mylooks')||'null');if(m)DECKS['My Looks']=m}catch(e){}
$('#saveLook').onclick=()=>{if(!hasImg)return;const D=DECKS['My Looks']=DECKS['My Looks']||{},n=($('#lookName').value||'My look '+(Object.keys(D).length+1)).trim(),p={};for(const k in DEF)if(S[k])p[k]=+(+S[k]).toFixed(3);D[n]=p;
 try{localStorage.setItem('mylooks',JSON.stringify(D))}catch(e){}deck='My Looks';cp=p;renderDeck();buildThumbs();slate('Saved: '+n);$('#lookName').value=''};
document.querySelectorAll('[data-ar]').forEach(b=>b.onclick=()=>{if(!hasImg)return;draw();const[a,c]=b.dataset.ar.split(':').map(Number),tw=a>c?1920:1080,th=Math.round(tw*c/a),o=document.createElement('canvas');o.width=tw;o.height=th;const s=Math.max(tw/W,th/H),sw=tw/s,sh=th/s;
 o.getContext('2d').drawImage(cv,(W-sw)/2,(H-sh)/2,sw,sh,0,0,tw,th);o.toBlob(q=>dl(q,fname('_'+a+'x'+c+'.png')))});
// ================= MORE FUN ZONE ACTIVITIES =================
const photoIn=(el,cb)=>{const l=document.createElement('label');l.className='btn';l.innerHTML='Use my photo<input type="file" accept="image/*" hidden>';l.querySelector('input').onchange=async e=>{if(e.target.files[0])cb(await createImageBitmap(e.target.files[0]))};return l};
const cover=(g,s,w,h)=>{const k=Math.max(w/s.width,h/s.height);g.drawImage(s,(w-s.width*k)/2,(h-s.height*k)/2,s.width*k,s.height*k)};
function FUN2(){return[['Disposable Camera','Shoot 24, develop later',dispoCam],['Subtitle This','Cinema subtitles on a photo',subtitleToy],['Composition Coach','Grids and crops that teach framing',compoToy],['Which Camera Suits Me?','A three-question match',camQuiz],['Palette Wallpaper','Turn a photo into a wallpaper',palWall],['Roll the Credits','Your session as end titles',el=>{rollCredits();setTimeout(funHome,0)}]]}
function dispoCam(el){el.innerHTML='<video id="dv" playsinline muted></video><p id="dn">24 shots left</p><div><button class="btn" id="dsh">Click</button><button class="btn" id="ddv">Develop roll</button></div><div id="dg" class="gg"></div><p class="note">You cannot peek. Photos stay on this phone until you save them.</p>';
 const v=el.querySelector('#dv'),n=el.querySelector('#dn'),shots=[];let st=null;
 navigator.mediaDevices&&navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}}).then(s=>{st=s;v.srcObject=s;v.play()}).catch(()=>n.textContent='Camera is not available here');
 el.querySelector('#dsh').onclick=()=>{if(!v.videoWidth||shots.length>=24)return;const c=document.createElement('canvas');c.width=900;c.height=Math.round(900*v.videoHeight/v.videoWidth);c.getContext('2d').drawImage(v,0,0,c.width,c.height);shots.push(c);if(shots.length%6==0)tixAdd('dispo',1,4);navigator.vibrate&&navigator.vibrate(20);n.textContent=(24-shots.length)+' shots left';v.style.opacity=.2;setTimeout(()=>v.style.opacity=1,120)};
 el.querySelector('#ddv').onclick=()=>{const F=['sepia(.35) saturate(1.4) contrast(1.1)','saturate(1.6) hue-rotate(-10deg)','contrast(1.2) brightness(1.05) saturate(1.2)','sepia(.2) hue-rotate(15deg) saturate(1.3)'],d=new Date(),ds="'"+String(d.getFullYear()).slice(2)+' '+(d.getMonth()+1)+' '+d.getDate(),G=el.querySelector('#dg');G.innerHTML='';
  shots.forEach(s=>{const c=document.createElement('canvas');c.width=s.width;c.height=s.height;const g=c.getContext('2d');g.filter=F[Math.floor(Math.random()*F.length)];g.drawImage(s,0,0);g.filter='none';
   const side=Math.random()<.5?0:c.width,lk=g.createRadialGradient(side,c.height*Math.random(),0,side,c.height*.5,c.width*.7);lk.addColorStop(0,'rgba(255,120,40,.7)');lk.addColorStop(1,'rgba(255,60,0,0)');g.globalCompositeOperation='screen';g.fillStyle=lk;g.fillRect(0,0,c.width,c.height);g.globalCompositeOperation='source-over';
   for(let i=0;i<9000;i++){g.fillStyle=Math.random()<.5?'rgba(255,255,255,.08)':'rgba(0,0,0,.1)';g.fillRect(Math.random()*c.width,Math.random()*c.height,2,2)}g.fillStyle='#ff9a2e';g.font='700 34px monospace';g.textAlign='right';g.fillText(ds,c.width-24,c.height-24);
   const b=document.createElement('button');b.appendChild(c);b.onclick=()=>c.toBlob(q=>dl(q,'disposable_'+Date.now()+'.jpg'));G.append(b)});n.textContent=shots.length?'Developed. Tap a photo to save it.':'Take a few shots first'};
 return()=>{if(st)st.getTracks().forEach(t=>t.stop())}}
function subtitleToy(el){el.innerHTML='<canvas id="sc" width="640" height="360"></canvas><input id="stx" class="inp" value="Some stories begin on a quiet street."><div></div><button class="btn" id="ssv">Save</button>';let img=null;
 const c=el.querySelector('#sc'),g=c.getContext('2d'),paint=()=>{g.fillStyle='#000';g.fillRect(0,0,640,360);cover(g,img||sampleCanvas(),640,360);g.fillStyle='#000';g.fillRect(0,0,640,44);g.fillRect(0,316,640,44);g.font='700 22px system-ui';g.textAlign='center';g.lineWidth=5;g.strokeStyle='#000';g.fillStyle='#fff';
  const t=el.querySelector('#stx').value,w=t.split(' ');let a=t,b='';if(g.measureText(t).width>600){const m=Math.ceil(w.length/2);a=w.slice(0,m).join(' ');b=w.slice(m).join(' ')}
  const ln=(x,y)=>{g.strokeText(x,320,y);g.fillText(x,320,y)};if(b){ln(a,274);ln(b,302)}else ln(a,300)};
 el.children[2].replaceWith(photoIn(el,i=>{img=i;paint()}));el.querySelector('#stx').oninput=paint;el.querySelector('#ssv').onclick=()=>c.toBlob(b=>dl(b,'subtitle.png'));paint()}
function compoToy(el){el.innerHTML='<canvas id="cc" width="600" height="400"></canvas><div id="cg"></div><div id="ca2"></div>';const G=['Thirds','Golden ratio','Diagonals','Center','None'],A=[['Free',0],['1:1',1],['4:5',.8],['16:9',16/9],['9:16',9/16]];let img=null,gi=0,ai=0;
 const c=el.querySelector('#cc'),g=c.getContext('2d'),paint=()=>{cover(g,img||sampleCanvas(),600,400);let x=0,y=0,w=600,h=400;const r=A[ai][1];if(r){w=600;h=w/r;if(h>400){h=400;w=h*r}x=(600-w)/2;y=(400-h)/2;g.fillStyle='rgba(0,0,0,.65)';g.fillRect(0,0,600,y);g.fillRect(0,y+h,600,400-y-h);g.fillRect(0,y,x,h);g.fillRect(x+w,y,600-x-w,h)}
  g.strokeStyle='rgba(255,255,255,.85)';g.lineWidth=1.5;g.beginPath();const L=(a,b,c2,d)=>{g.moveTo(a,b);g.lineTo(c2,d)},V=f=>L(x+w*f,y,x+w*f,y+h),Hh=f=>L(x,y+h*f,x+w,y+h*f);
  if(gi==0){V(1/3);V(2/3);Hh(1/3);Hh(2/3)}if(gi==1){V(.382);V(.618);Hh(.382);Hh(.618)}if(gi==2){L(x,y,x+w,y+h);L(x+w,y,x,y+h)}if(gi==3){V(.5);Hh(.5)}g.stroke()};
 const chips=(t,list,cur,set)=>{t.innerHTML='';list.forEach((o,i)=>{const b=document.createElement('button');b.className='btn'+(i===cur()?' on':'');b.textContent=Array.isArray(o)?o[0]:o;b.onclick=()=>{set(i);paint();chips(t,list,cur,set)};t.append(b)})};
 chips(el.querySelector('#cg'),G,()=>gi,i=>gi=i);chips(el.querySelector('#ca2'),A,()=>ai,i=>ai=i);el.append(photoIn(el,i=>{img=i;paint()}));paint()}
function camQuiz(el){const Q=[['Your budget (US dollars)?',[['Under 1,200',1200],['Up to 2,200',2200],['No limit',99999]]],['What will you shoot most?',[['Travel',0],['Vlogs and video',1],['Portraits',2],['Street',3]]],['How much do you want to carry?',[['Tiny, pocketable (under 450 g)',450],['Medium (under 650 g)',650],['I do not mind (any)',9999]]]],a=[];
 const next=()=>{if(a.length==3){const[b,u,wt]=a;let L=CAMS.filter(c=>c[6]<=b&&c[5]<=wt);if(L.length<3)L=CAMS.filter(c=>c[6]<=b*1.3&&c[5]<=wt*1.2);
  const sc=c=>u==0?c[4]*1.2-c[5]/120:u==1?(/6K|8K|120/.test(c[3])?4:1)+c[4]:u==2?(c[1]=='Full-frame'?4:1)+c[2]/30:-c[5]/90+(/X100/.test(c[0])?5:0);L=L.sort((x,y)=>sc(y)-sc(x)).slice(0,3);
  el.innerHTML='<h2>Your top picks</h2>'+(L.length?L.map(c=>'<div class="pick"><b>'+c[0]+'</b><span>'+c[1]+' \u00b7 '+c[2]+' MP \u00b7 '+c[3]+' \u00b7 '+c[5]+' g \u00b7 about $'+c[6]+' at launch</span></div>').join(''):'<p>Nothing fits those limits. Try a bigger budget.</p>')+'<button class="btn" id="qr">Again</button><p class="note">Based on specs and US launch prices from memory. Check live prices before buying.</p>';el.querySelector('#qr').onclick=()=>{a.length=0;next()};return}
  const q=Q[a.length];el.innerHTML='<p>Question '+(a.length+1)+' of 3</p><h2>'+q[0]+'</h2>'+q[1].map((o,i)=>'<button class="btn qo" data-i="'+i+'">'+o[0]+'</button>').join('');el.querySelectorAll('.qo').forEach(b=>b.onclick=()=>{a.push(q[1][+b.dataset.i][1]);next()})};next()}
function palWall(el){el.innerHTML='<canvas id="pw" width="270" height="585"></canvas><div id="pal2"></div><button class="btn" id="pws">Save wallpaper</button>';let img=null;const c=el.querySelector('#pw'),g=c.getContext('2d');
 const go=()=>{const s=img||sampleCanvas(),t=document.createElement('canvas');t.width=t.height=64;const tg=t.getContext('2d');tg.drawImage(s,0,0,64,64);const d=tg.getImageData(0,0,64,64).data,q={};for(let i=0;i<d.length;i+=4){const k=(d[i]>>5)+','+(d[i+1]>>5)+','+(d[i+2]>>5);q[k]=(q[k]||0)+1}
  const cols=Object.entries(q).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k])=>{const p=k.split(',').map(v=>v*36+18);return'rgb('+p+')'});
  g.fillStyle=cols[cols.length-1];g.fillRect(0,0,270,585);[[.2,.2,.7],[.85,.45,.6],[.25,.75,.65],[.8,.9,.5]].forEach(([x,y,r],i)=>{const gr=g.createRadialGradient(x*270,y*585,0,x*270,y*585,r*400);gr.addColorStop(0,cols[i%cols.length]);gr.addColorStop(1,'rgba(0,0,0,0)');g.fillStyle=gr;g.fillRect(0,0,270,585)});
  el.querySelector('#pal2').innerHTML=cols.map(x=>'<i style="background:'+x+'"></i>').join('')};
 el.insertBefore(photoIn(el,i=>{img=i;go()}),el.querySelector('#pws'));el.querySelector('#pws').onclick=()=>{const o=document.createElement('canvas');o.width=1080;o.height=2340;o.getContext('2d').drawImage(c,0,0,1080,2340);o.toBlob(b=>dl(b,'wallpaper.png'))};go()}

// ================= TICKET SYSTEM: photos earn tickets, activities spend them =================
const tixGet=()=>lsn('tix'),tixSet=n=>{try{localStorage.setItem('tix',n)}catch(e){}tixRender()};
funB.id='funBtn';
(function(){try{const first=localStorage.getItem('tix')===null;if(first)localStorage.setItem('tix',3);const d=new Date().toDateString();if(localStorage.getItem('tixDay')!==d){localStorage.setItem('tixDay',d);if(!first)localStorage.setItem('tix',tixGet()+1);localStorage.removeItem('tixCaps')}}catch(e){}})();
function tixAdd(src,n,cap){n=n||1;cap=cap||5;let c={};try{c=JSON.parse(localStorage.getItem('tixCaps')||'{}')}catch(e){}if((c[src]||0)>=cap)return false;c[src]=(c[src]||0)+1;
 try{localStorage.setItem('tixCaps',JSON.stringify(c));const life=lsn('tixLife')+1;localStorage.setItem('tixLife',life);if(life%10===0){n+=3;toast('Ten photos! Bonus tickets')}}catch(e){}tixSet(tixGet()+n);tixFly(n);return true}
function tixFly(n){const f=document.createElement('div');f.className='tfly';f.innerHTML='<i></i>+'+n+' ticket'+(n>1?'s':'');document.body.append(f);const r=funB.getBoundingClientRect(),a=f.animate([{transform:'translate(-50%,0) scale(.6)',opacity:0},{transform:'translate(-50%,-30px) scale(1)',opacity:1,offset:.25},{transform:'translate('+(r.left+r.width/2-innerWidth/2)+'px,'+(r.top-innerHeight+150)+'px) scale(.4)',opacity:.2}],{duration:1100,easing:'cubic-bezier(.5,0,.3,1)'});
 a.onfinish=()=>{f.remove();funB.classList.remove('pop');void funB.offsetWidth;funB.classList.add('pop')};navigator.vibrate&&navigator.vibrate(12)}
function tixRender(){const n=tixGet();let b=funB.querySelector('.bdg');if(!b){b=document.createElement('span');b.className='bdg';funB.append(b)}b.textContent=n;const s=$('#stubs'),su=$('.sub');
 if(s)s.innerHTML='<i></i>'.repeat(Math.min(n,12))+(n>12?'<em>+'+(n-12)+'</em>':'');if(su)su.textContent=n?'Each activity costs one ticket. Earn more by shooting, editing and saving photos.':'Out of tickets. Shoot, edit or save a photo to earn more.'}
function tixSpend(b){if(tixGet()<=0){b.classList.remove('shake');void b.offsetWidth;b.classList.add('shake');navigator.vibrate&&navigator.vibrate([40,40,40]);toast('No tickets left. Shoot or save a photo to earn more');tixRender();return false}
 tixSet(tixGet()-1);b.classList.add('tear');setTimeout(()=>b.classList.remove('tear'),700);navigator.vibrate&&navigator.vibrate(18);return true}
tixRender();

// ================= OPENING v2: aperture blades close, Launch, shutter, blades fly open, ripples =================
(function(){let skip=false;try{if(sessionStorage.getItem('skipIntro')){sessionStorage.removeItem('skipIntro');skip=true}}catch(e){}
 if(skip||!$('#empty')||matchMedia('(prefers-reduced-motion:reduce)').matches)return;
 let quick=false;const day=new Date().toDateString();try{quick=localStorage.getItem('apDay')===day;localStorage.setItem('apDay',day)}catch(e){}
 const B=document.body;B.classList.add('pre');
 const ap=mk('div','ap','<svg id="apS"><defs><radialGradient id="apG"><stop offset="0" stop-color="#34343a"/><stop offset="1" stop-color="#0b0b0d"/></radialGradient></defs><path id="apP" fill="url(#apG)" fill-rule="evenodd"/><path id="apL" stroke="rgba(255,255,255,.14)" stroke-width="1" fill="none"/></svg><div id="lens"></div><button id="launchB">LAUNCH</button><p id="apHint">TAP ANYWHERE</p><button id="apSkip">SKIP</button>');
 const P=$('#apP'),Lp=$('#apL'),lens=$('#lens'),lb=$('#launchB'),N=8;let w,h,cx,cy,Rf,Rs,R,st='open';
 const geo=()=>{w=innerWidth;h=innerHeight;cx=w/2;cy=h/2;Rf=Math.hypot(w,h)/2*1.15;Rs=Math.min(w,h)*.17;$('#apS').setAttribute('viewBox','0 0 '+w+' '+h)};geo();addEventListener('resize',geo);R=Rf;
 const draw2=()=>{const a=(1-Math.min(1,R/Rf))*.55,rv=R/Math.cos(Math.PI/N),pts=[];for(let i=0;i<N;i++){const t=a+i*2*Math.PI/N;pts.push([cx+Math.cos(t)*rv,cy+Math.sin(t)*rv])}
  P.setAttribute('d','M0 0H'+w+'V'+h+'H0Z'+pts.map((p,i)=>(i?'L':'M')+p[0].toFixed(1)+' '+p[1].toFixed(1)).join('')+'Z');let l='';
  pts.forEach((p,i)=>{const q=pts[(i+1)%N],dx=q[0]-p[0],dy=q[1]-p[1],m=Math.hypot(dx,dy)||1,ux=dx/m*Rf,uy=dy/m*Rf;l+='M'+(p[0]-ux).toFixed(1)+' '+(p[1]-uy).toFixed(1)+'L'+(q[0]+ux).toFixed(1)+' '+(q[1]+uy).toFixed(1)});Lp.setAttribute('d',l);
  lens.style.display=R<Rs*2?'block':'none';lens.style.width=lens.style.height=(2*R*.985)+'px'};draw2();
 const eo=p=>1-Math.pow(1-p,3),ei=p=>p*p*p,eb=p=>1-Math.pow(1-p,4);
 const tw=(to,ms,ease,done)=>{const f0=R,t0=performance.now();(function f(now){const p=Math.min(1,(now-t0)/ms);R=f0+(to-f0)*ease(p);draw2();if(p<1)requestAnimationFrame(f);else done&&done()})(t0)};
 function ripple(){const W2=innerWidth,H2=innerHeight,sp=Math.hypot(W2,H2)/1500;
  for(let i=0;i<4;i++){const r=document.createElement('div');r.className='rip';document.body.append(r);r.animate([{transform:'translate(-50%,-50%) scale(0)',opacity:.75},{transform:'translate(-50%,-50%) scale('+(Math.hypot(W2,H2)/60)+')',opacity:0}],{duration:1500,delay:i*140,easing:'cubic-bezier(.2,.6,.3,1)'}).onfinish=()=>r.remove()}
  const go=(sel,kf)=>document.querySelectorAll(sel).forEach(el=>{const b=el.getBoundingClientRect(),d=Math.hypot(b.left+b.width/2-W2/2,b.top+b.height/2-H2/2);el.style.animation=kf+' .95s cubic-bezier(.2,.7,.3,1) '+Math.round(d/sp*.55)+'ms both';setTimeout(()=>el.style.animation='',2600)});
  ['#mood','#dot','#mid','#greet','#vfl','#vfr','#top .rb','#top .logo','#empty .big'].forEach(s=>go(s,'wob2'));['#reel','#reelLit'].forEach(s=>go(s,'wob3'))}
 function launch(){if(st==='snap'||st==='done')return;st='snap';lb.classList.remove('on');ap.style.pointerEvents='none';navigator.vibrate&&navigator.vibrate([25,30,40]);
  try{if($('#snd').checked){AC=AC||new AudioContext();const b=AC.createBuffer(1,6000,44100),d=b.getChannelData(0);for(let i=0;i<6000;i++)d[i]=(Math.random()*2-1)*Math.exp(-i/700);const s=AC.createBufferSource();s.buffer=b;s.connect(AC.destination);s.start()}}catch(e){}
  tw(0,110,ei,()=>setTimeout(()=>{B.classList.remove('pre','lensed');ripple();st='done';tw(Rf,720,eb,()=>ap.remove())},80))}
 ap.addEventListener('click',e=>{if(e.target.id==='apSkip'){st='done';B.classList.remove('pre','lensed');ap.remove();return}if(st==='open'){st='closing';B.classList.add('lensed');const hn=$('#apHint');if(hn)hn.remove();navigator.vibrate&&navigator.vibrate(10);tw(Rs,900,eo,()=>{st='closed';lb.classList.add('on')})}});
 lb.onclick=e=>{e.stopPropagation();launch()};
 if(quick){const hn=$('#apHint');if(hn)hn.remove();R=Rs;draw2();B.classList.add('lensed');st='closed';setTimeout(launch,350)}
})();

// ================= MASTERS: famous cinema colour palettes (approximations in the style of well-known films) =================
const MASTERS=[
['Teal & Orange','Modern blockbuster style',['#0f4c5c','#1f7a8c','#c9a27e','#e36414','#fb8b24'],{con:.25,sat:.1},{lift:[.5,.55],gain:[.08,.5]}],
['Neon Noir','Blade Runner 2049 style',['#0b1d2a','#1b6a7a','#c4577a','#e8742a','#f2b36b'],{con:.2,sat:.15,halo:.4,grain:.2,vig:.3,temp:.1},{lift:[.52,.45],gain:[.07,.6]}],
['Digital Rain','The Matrix style',['#031a0c','#0c4a24','#2f9e57','#9fe3b0','#d8f5de'],{con:.3,sat:-.1,tint:-.4,grain:.2,vig:.25},{lift:[.38,.35],gam:[.33,.5]}],
['Pastel Symmetry','The Grand Budapest Hotel style',['#f4b6c8','#ffd9e3','#e5a0b4','#8fb8c9','#f7e7c9'],{sat:-.1,con:-.1,ev:.15,temp:.1,tint:.2,sh:.3},{gain:[.93,.25],lift:[.95,.2]}],
['Desert Fury','Mad Max: Fury Road style',['#06303a','#0f6a7a','#c96a1e','#f08a24','#ffd08a'],{con:.35,sat:.4,temp:.5,vig:.3},{lift:[.5,.4],gain:[.08,.55]}],
['Cold Steel','Fincher thriller style',['#0b0f0c','#2b3a2c','#58664a','#a8a37a','#d6d0b4'],{con:.35,sat:-.45,temp:-.1,tint:-.2,vig:.4,grain:.35},{gam:[.2,.3],lift:[.4,.25]}],
['Storybook Warm','Amelie style',['#1f4a2a','#4a8a3a','#c4452c','#e8a64a','#f6e0b0'],{sat:.3,con:.15,temp:.4,tint:-.1},{lift:[.33,.3],gain:[.04,.5]}],
['Bleach Wartime','Saving Private Ryan style',['#2a2d2e','#4b5254','#7b8587','#aeb4b0','#d9d8cf'],{con:.45,sat:-.55,hi:-.15,grain:.5,gwv:.2},{}],
['Golden Epic','Gladiator style',['#2a1a0a','#6a4a1a','#b88a3a','#e2c070','#f3e2b0'],{sat:-.2,temp:.35,con:.2,grain:.3,vig:.3},{gain:[.11,.35]}],
['Blue Violet','Moonlight style',['#101a3a','#2a3f9a','#6a5acd','#c05a9a','#f08ab0'],{sat:.2,con:.1,temp:-.2,tint:.3},{lift:[.66,.4],gain:[.85,.4]}],
['Ochre Dust','Dune style',['#2a2016','#6a5a3a','#b08a50','#d8b878','#efe0bd'],{sat:-.3,temp:.4,con:.15,ev:.1,grain:.2},{gain:[.1,.3]}],
['Pink Teal Night','Drive style',['#05202a','#0f6a78','#d04a8a','#f27ab0','#ffd0e0'],{sat:.2,con:.2,halo:.4,vig:.3,tint:.3},{lift:[.5,.45],gain:[.9,.5]}]];
function applyMaster(m){if(!hasImg){toast('Open a photo first');return}cp=null;setAll({...m[3]});const ws=['lift','gam','gain'];
 ws.forEach(w=>{const v=m[4][w];Object.assign(WS[w],v?{x:Math.cos(v[0]*6.2832)*v[1],y:Math.sin(v[0]*6.2832)*v[1],l:0}:{x:0,y:0,l:0});WH[w]=wheelVec(w)});drawWheels();['wlLift','wlGam','wlGain'].forEach((id,i)=>$('#'+id).value=0);draw();slate('Master: '+m[0]);snap()}
MASTERS.forEach(m=>{const b=document.createElement('button');b.className='mcard';b.innerHTML='<span class="sw">'+m[2].map(c=>'<i style="background:'+c+'"></i>').join('')+'</span><b>'+m[0]+'</b><small>'+m[1]+'</small>';b.onclick=()=>applyMaster(m);$('#mstrs').append(b)});
// ================= LOOK CARD: a collectible card for your grade =================
$('#lookCard').onclick=()=>{if(!hasImg)return;draw();const c=document.createElement('canvas');c.width=720;c.height=1080;const g=c.getContext('2d'),
 nz=Object.keys(DEF).filter(k=>S[k]&&Math.abs(S[k])>.02).length,rar=nz>=12?['LEGENDARY','#ffd36b']:nz>=7?['RARE','#6aa8ff']:['COMMON','#c8c8cc'],nm=lookName()==='custom'?'My Custom Grade':lookName();
 g.fillStyle='#0c0c0e';g.fillRect(0,0,720,1080);const fg=g.createLinearGradient(0,0,720,1080);fg.addColorStop(0,rar[1]);fg.addColorStop(.5,'#ffffff');fg.addColorStop(1,rar[1]);g.strokeStyle=fg;g.lineWidth=10;g.strokeRect(12,12,696,1056);
 g.save();g.beginPath();g.roundRect(48,48,624,624,22);g.clip();const k=Math.max(624/W,624/H);g.drawImage(cv,48+(624-W*k)/2,48+(624-H*k)/2,W*k,H*k);g.restore();
 g.textAlign='left';g.fillStyle='#fff';let fs=46;g.font='700 '+fs+'px Bricolage Grotesque,system-ui,sans-serif';while(g.measureText(nm).width>620&&fs>26){fs-=2;g.font='700 '+fs+'px Bricolage Grotesque,system-ui,sans-serif'}g.fillText(nm,48,742);
 g.fillStyle=rar[1];g.font='700 20px JetBrains Mono,monospace';g.fillText(rar[0]+'  \u00b7  '+nz+' ADJUSTMENTS',48,782);
 const t=document.createElement('canvas');t.width=t.height=48;const tg=t.getContext('2d');tg.drawImage(cv,0,0,48,48);const d=tg.getImageData(0,0,48,48).data,q={};for(let i=0;i<d.length;i+=4){const key=(d[i]>>5)+','+(d[i+1]>>5)+','+(d[i+2]>>5);q[key]=(q[key]||0)+1}
 Object.entries(q).sort((a,b)=>b[1]-a[1]).slice(0,5).forEach(([key],i)=>{const p=key.split(',').map(v=>v*36+18);g.fillStyle='rgb('+p+')';g.beginPath();g.arc(74+i*74,836,28,0,7);g.fill();g.strokeStyle='rgba(255,255,255,.4)';g.lineWidth=2;g.stroke()});
 [['WARMTH',S.temp],['CONTRAST',S.con],['GRAIN',S.grain],['COLOUR',S.sat]].forEach(([n,v],i)=>{v=v||0;const y=915+i*32;g.fillStyle='#8d8d93';g.font='500 16px JetBrains Mono,monospace';g.fillText(n,48,y);for(let j=0;j<10;j++){g.fillStyle=j<Math.round(Math.abs(v)*10)?'#fff':'#2a2a2e';g.fillRect(220+j*32,y-13,26,13)}g.fillStyle='#8d8d93';g.fillText((v>0?'+':v<0?'-':'')+Math.abs(Math.round(v*100)),560,y)});
 let h=0;const ss=state();for(let i=0;i<ss.length;i++)h=(h*31+ss.charCodeAt(i))>>>0;g.fillStyle='#6d6d73';g.font='500 14px JetBrains Mono,monospace';g.fillText('LUMINA STUDIO  \u00b7  '+new Date().toLocaleDateString()+'  \u00b7  No. '+String(h%1000).padStart(3,'0'),48,1052);
 tixAdd('card',1,3);c.toBlob(b=>dl(b,fname('_card.png')))};

// ================= PRIVACY AND STORAGE =================
$('#privB').onclick=async()=>{let u='';try{const e=await navigator.storage.estimate();u=(e.usage/1048576).toFixed(1)+' MB used by this app on this device'}catch(e){u='Storage size is not available here'}
 const old=$('#priv');if(old)old.remove();const m=mk('div','priv','<div class="pc"><h2>Privacy and storage</h2><p>Your photos are processed on this phone and are never uploaded. The app keeps only your last photo and edit (for Continue), your tickets, saved looks and a few settings, all inside this browser on this device.</p><p>The internet is used only for fonts, for the AI tools when you first use them, and for price links you tap.</p><p><b>'+u+'</b></p><button class="btn" id="pc1">Clear last photo</button><button class="btn" id="pc2">Clear tickets and saved looks</button><button class="btn" id="pc3">Clear everything</button><button class="btn" id="pcx">Close</button></div>');
 const kill=()=>{try{indexedDB.deleteDatabase('lumina')}catch(e){}};
 $('#pc1').onclick=()=>{kill();toast('Last photo cleared');m.remove()};
 $('#pc2').onclick=()=>{['tix','tixDay','tixCaps','tixLife','mylooks','photos','take','apDay'].forEach(k=>{try{localStorage.removeItem(k)}catch(e){}});toast('Cleared. Reload to see it');m.remove()};
 $('#pc3').onclick=async()=>{try{localStorage.clear()}catch(e){}kill();try{(await caches.keys()).forEach(k=>caches.delete(k))}catch(e){}location.reload()};
 $('#pcx').onclick=()=>m.remove()};

// ================= SHARED HELPERS =================
const small=(im,m)=>{const k=Math.min(1,m/Math.max(im.width,im.height)),c=document.createElement('canvas');c.width=Math.round(im.width*k);c.height=Math.round(im.height*k);c.getContext('2d').drawImage(im,0,0,c.width,c.height);return c};
const SEGOPT=d=>({baseOptions:{delegate:d,modelAssetPath:'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite'},runningMode:'IMAGE',outputCategoryMask:true,outputConfidenceMasks:false});
// ================= SHATTER: the page cracks, breaks into glass shards and falls away =================
function shatter(){const w=innerWidth,h=innerHeight,r=funB.getBoundingClientRect(),ix=r.left+r.width/2,iy=r.top+r.height/2,C=5,Rw=7,P=[],hasP=document.body.classList.contains('has'),tod=$('#tod'),
 bg=hasP?'url('+cv.toDataURL('image/jpeg',.55)+')':(tod?getComputedStyle(tod).backgroundImage:'linear-gradient(#222,#000)'),cr=cv.getBoundingClientRect();
 const box=document.createElement('div');box.id='shat';document.body.append(box);
 for(let y=0;y<=Rw;y++){P[y]=[];for(let x=0;x<=C;x++)P[y][x]=[(x+((x>0&&x<C)?(Math.random()-.5)*.6:0))/C*w,(y+((y>0&&y<Rw)?(Math.random()-.5)*.6:0))/Rw*h]}
 const sh=[];for(let y=0;y<Rw;y++)for(let x=0;x<C;x++){const a=P[y][x],b=P[y][x+1],c=P[y+1][x+1],d=P[y+1][x];sh.push([a,b,c],[a,c,d])}
 sh.forEach(t=>{const xs=t.map(p=>p[0]),ys=t.map(p=>p[1]),l=Math.min(...xs),tp=Math.min(...ys),wd=Math.max(...xs)-l,ht=Math.max(...ys)-tp,e=document.createElement('div');e.className='shard';
  Object.assign(e.style,{left:l+'px',top:tp+'px',width:wd+'px',height:ht+'px',clipPath:'polygon('+t.map(p=>(p[0]-l)+'px '+(p[1]-tp)+'px').join(',')+')',
   backgroundImage:'linear-gradient(115deg,rgba(255,255,255,'+(.1+Math.random()*.2).toFixed(2)+'),transparent 45%),'+bg,
   backgroundSize:'auto,'+(hasP?cr.width+'px '+cr.height+'px':w+'px '+h+'px'),backgroundPosition:'0 0,'+((hasP?cr.left:0)-l)+'px '+((hasP?cr.top:0)-tp)+'px',backgroundColor:'#050505',backgroundRepeat:'no-repeat'});
  const cx=(xs[0]+xs[1]+xs[2])/3,cy=(ys[0]+ys[1]+ys[2])/3,dist=Math.hypot(cx-ix,cy-iy);e.dataset.d=dist;box.append(e)});
 const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 '+w+' '+h);svg.style.cssText='position:absolute;inset:0;width:100%;height:100%';box.append(svg);
 for(let i=0;i<16;i++){const an=Math.random()*6.283,len=Math.max(w,h)*(.25+Math.random()*.5);let d='M'+ix+' '+iy,px=ix,py=iy;for(let k=1;k<=4;k++){px+=Math.cos(an+(Math.random()-.5)*.4)*len/4;py+=Math.sin(an+(Math.random()-.5)*.4)*len/4;d+='L'+px.toFixed(0)+' '+py.toFixed(0)}
  const p=document.createElementNS('http://www.w3.org/2000/svg','path');p.setAttribute('d',d);p.setAttribute('stroke','rgba(255,255,255,.85)');p.setAttribute('stroke-width','1.5');p.setAttribute('fill','none');p.setAttribute('stroke-dasharray',len*1.3);svg.append(p);p.animate([{strokeDashoffset:len*1.3},{strokeDashoffset:0}],{duration:260,fill:'forwards',easing:'ease-out'})}
 navigator.vibrate&&navigator.vibrate([20,40,60,30,20]);
 try{if($('#snd').checked){AC=AC||new AudioContext();const b=AC.createBuffer(1,16000,44100),d=b.getChannelData(0);for(let i=0;i<16000;i++)d[i]=(Math.random()*2-1)*Math.exp(-i/2500);const s=AC.createBufferSource();s.buffer=b;s.connect(AC.destination);s.start()}}catch(e){}
 setTimeout(()=>{svg.remove();box.querySelectorAll('.shard').forEach(e=>{const d=+e.dataset.d,dx=(Math.random()-.5)*180,rot=(Math.random()-.5)*340;e.animate([{transform:'translate(0,0) rotate(0)'},{transform:'translate('+dx+'px,'+(h*1.4+Math.random()*240)+'px) rotate('+rot+'deg)'}],{duration:900+Math.random()*500,delay:d*.55,easing:'cubic-bezier(.55,0,.9,.4)',fill:'forwards'})});setTimeout(()=>box.remove(),2300)},300)}
// ================= LIVE MONITOR HUD =================
let camIdx=-1,camList=[],lrec=null,lch=[];
async function camStream(){const co=matchMedia('(pointer:coarse)').matches,base={width:{ideal:co?960:1280},height:{ideal:co?540:720}};
 try{return await navigator.mediaDevices.getUserMedia({video:camList[camIdx]?{...base,deviceId:{exact:camList[camIdx]}}:{...base,facingMode:'environment'},audio:false})}catch(e){return await navigator.mediaDevices.getUserMedia({video:base,audio:false})}}
const lh=mk('div','lhud','<button data-l="flip">Flip</button><button data-l="grid">Grid</button><button data-l="zeb">Zebra</button><button data-l="fc">False</button><button data-l="rec">Rec</button><span id="lfps"></span>');mk('div','lgrid');
lh.onclick=async e=>{const b=e.target.closest('button');if(!b)return;const k=b.dataset.l;
 if(k==='grid'){document.body.classList.toggle('lgrid');b.classList.toggle('on')}else if(k==='zeb'){S.zeb=S.zeb?0:1;b.classList.toggle('on')}else if(k==='fc'){S.fc=S.fc?0:1;b.classList.toggle('on')}
 else if(k==='flip'){try{camList=(await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput').map(d=>d.deviceId);camIdx=(camIdx+1)%Math.max(1,camList.length);await live(false);await live(true)}catch(x){toast('Cannot switch camera')}}
 else if(k==='rec'){if(lrec){lrec.stop();lrec=null;b.classList.remove('on');b.textContent='Rec'}else if(window.MediaRecorder){lch=[];lrec=new MediaRecorder(cv.captureStream(30),{mimeType:MediaRecorder.isTypeSupported('video/webm;codecs=vp9')?'video/webm;codecs=vp9':'video/webm'});lrec.ondataavailable=v=>lch.push(v.data);lrec.onstop=()=>dl(new Blob(lch,{type:'video/webm'}),'live_'+Date.now()+'.webm');lrec.start();b.classList.add('on');b.textContent='Stop'}}};
addEventListener('keydown',e=>{if(!liveOn)return;if(e.code==='Space'){e.preventDefault();sh.click()}else if(e.key==='g')document.body.classList.toggle('lgrid');else if(e.key==='r')lh.querySelector('[data-l=rec]').click()});

// ================= UI ICING: ambilight edges, readable text, film roll counter, pinned tools =================
function edgeGlow(){if(!hasImg||document.hidden)return;const c=document.createElement('canvas');c.width=c.height=8;const g=c.getContext('2d',{willReadFrequently:true});g.drawImage(cv,0,0,8,8);const d=g.getImageData(0,0,8,8).data,s=document.body.style,
 av=f=>{let r=0,gg=0,b=0,n=0;for(let i=0;i<64;i++){if(!f(i%8,Math.floor(i/8)))continue;const k=i*4;r+=d[k];gg+=d[k+1];b+=d[k+2];n++}return'rgb('+Math.round(r/n)+','+Math.round(gg/n)+','+Math.round(b/n)+')'};
 s.setProperty('--at',av((x,y)=>y<2));s.setProperty('--ab',av((x,y)=>y>5));s.setProperty('--al',av(x=>x<2));s.setProperty('--ar',av(x=>x>5));
 let L=0,n=0;for(let i=0;i<64;i++){if(Math.floor(i/8)<5)continue;const k=i*4;L+=.2126*d[k]+.7152*d[k+1]+.0722*d[k+2];n++}document.body.classList.toggle('bright',L/n/255>.6)}
document.addEventListener('pointerup',()=>setTimeout(edgeGlow,200));
// film roll counter
const rollEl=document.createElement('div');rollEl.id='roll';homeB.after(rollEl);let rollN=lsn('roll')||1,rollF=lsn('frame');
function rollRender(){rollEl.innerHTML='<b>ROLL '+rollN+'</b> '+rollF+'/36<i style="width:'+(rollF/36*100)+'%"></i>'}rollRender();
function rollAdvance(){rollF++;if(rollF<36){try{localStorage.setItem('frame',rollF)}catch(e){}rollRender();return}rollF=36;let f=36;
 const t=setInterval(()=>{f=Math.max(0,f-3);rollEl.innerHTML='<b>REWIND</b> '+f+'/36<i style="width:'+(f/36*100)+'%"></i>';if(!f){clearInterval(t);rollN++;rollF=0;try{localStorage.setItem('roll',rollN);localStorage.setItem('frame',0)}catch(e){}rollRender();tixSet(tixGet()+3);tixFly(3);slate('Roll '+(rollN-1)+' finished')}},70)}
// pinned tools: long-press a tool on the ring
const ORIG=TOOLS.map(t=>t[0]);let pins=new Set();try{pins=new Set(JSON.parse(localStorage.getItem('pins')||'[]'))}catch(e){}
function rebuildRing(){TOOLS.sort((a,b)=>(pins.has(b[0])-pins.has(a[0]))||(ORIG.indexOf(a[0])-ORIG.indexOf(b[0])));barrel.innerHTML='';
 TOOLS.forEach(([k,n],i)=>{const e=document.createElement('em');e.className='tl'+(pins.has(k)?' pin':'');e.dataset.k=k;e.textContent=n;e.style.transform='rotate('+i*STEP+'deg) translateY(calc(var(--R) * -1 + 46px)) translateX(-50%)';barrel.append(e)});
 engrave(barrel);rot=0;idx=0;place(0,0);lit(0);const t=document.body.dataset.tool;if(t)setTool(t)}
function togglePin(k){pins.has(k)?pins.delete(k):pins.add(k);try{localStorage.setItem('pins',JSON.stringify([...pins]))}catch(e){}navigator.vibrate&&navigator.vibrate([12,30,12]);toast(pins.has(k)?'Pinned to the front':'Unpinned');rebuildRing()}
let pinT=0;barrel.addEventListener('pointerdown',e=>{const em=e.target.closest('em.tl');if(!em)return;pinT=setTimeout(()=>{if(dr&&dr.m<4)togglePin(em.dataset.k)},650)});
['pointerup','pointercancel'].forEach(ev=>barrel.addEventListener(ev,()=>clearTimeout(pinT)));
rebuildRing();

// ================= UX TWEAKS =================
var WL=null;const lastToolSaved=()=>{try{return localStorage.getItem('lastTool')||'light'}catch(e){return'light'}};
{const _s2=setTool;setTool=function(t){if(t)try{localStorage.setItem('lastTool',t)}catch(e){}_s2(t)}}
// ================= CARDS: collectible look card + frame study =================
function analyze(img){const c=document.createElement('canvas');c.width=64;c.height=Math.max(1,Math.round(64*img.height/img.width));const g=c.getContext('2d',{willReadFrequently:true});g.drawImage(img,0,0,c.width,c.height);const d=g.getImageData(0,0,c.width,c.height).data,n=d.length/4,q={};let R=0,B=0,G=0,L=0,L2=0,S2=0;
 for(let i=0;i<d.length;i+=4){const r=d[i],gg=d[i+1],b=d[i+2],l=(.2126*r+.7152*gg+.0722*b)/255;R+=r;G+=gg;B+=b;L+=l;L2+=l*l;const mx=Math.max(r,gg,b),mn=Math.min(r,gg,b);S2+=mx?(mx-mn)/mx:0;const k=(r>>5)+','+(gg>>5)+','+(b>>5);q[k]=(q[k]||0)+1}
 const m=L/n;return{warm:(R-B)/n/255,bright:m,contrast:Math.min(1,Math.sqrt(Math.max(0,L2/n-m*m))*3),sat:S2/n,hues:Object.keys(q).length,cols:Object.entries(q).sort((a,b)=>b[1]-a[1]).slice(0,5).map(([k])=>k.split(',').map(v=>v*36+18))}}
function cardCanvas(o){const c=document.createElement('canvas');c.width=720;c.height=1080;const g=c.getContext('2d'),rar=o.rar;g.fillStyle='#0c0c0e';g.fillRect(0,0,720,1080);const fg=g.createLinearGradient(0,0,720,1080);fg.addColorStop(0,rar[1]);fg.addColorStop(.5,'#fff');fg.addColorStop(1,rar[1]);g.strokeStyle=fg;g.lineWidth=10;g.strokeRect(12,12,696,1056);
 g.save();g.beginPath();g.roundRect(48,48,624,624,22);g.clip();const k=Math.max(624/o.w,624/o.h);g.drawImage(o.src,48+(624-o.w*k)/2,48+(624-o.h*k)/2,o.w*k,o.h*k);g.restore();
 g.textAlign='left';g.fillStyle='#fff';let fs=46;g.font='700 '+fs+'px Bricolage Grotesque,system-ui,sans-serif';while(g.measureText(o.title).width>620&&fs>26){fs-=2;g.font='700 '+fs+'px Bricolage Grotesque,system-ui,sans-serif'}g.fillText(o.title,48,742);
 g.fillStyle=rar[1];g.font='700 18px JetBrains Mono,monospace';g.fillText(o.sub,48,782);o.cols.forEach((p,i)=>{g.fillStyle='rgb('+p+')';g.beginPath();g.arc(74+i*74,836,28,0,7);g.fill();g.strokeStyle='rgba(255,255,255,.4)';g.lineWidth=2;g.stroke()});
 o.stats.forEach(([n,v],i)=>{v=Math.max(-1,Math.min(1,v||0));const y=915+i*32;g.fillStyle='#8d8d93';g.font='500 16px JetBrains Mono,monospace';g.fillText(n,48,y);for(let j=0;j<10;j++){g.fillStyle=j<Math.round(Math.abs(v)*10)?'#fff':'#2a2a2e';g.fillRect(220+j*32,y-13,26,13)}g.fillStyle='#8d8d93';g.fillText((v>0?'+':v<0?'-':'')+Math.abs(Math.round(v*100)),560,y)});
 g.fillStyle='#6d6d73';g.font='500 14px JetBrains Mono,monospace';g.fillText(o.foot,48,1052);
 g.save();g.translate(548,112);g.rotate(.1);g.fillStyle='rgba(0,0,0,.72)';g.fillRect(-112,-36,224,72);g.strokeStyle=rar[1];g.lineWidth=3;g.strokeRect(-112,-36,224,72);g.fillStyle=rar[1];g.textAlign='center';g.font='700 15px JetBrains Mono,monospace';g.fillText('LUMINA EMPLOYEE',-12,-8);g.font='500 11px JetBrains Mono,monospace';g.fillText('APPROVED',-12,20);g.font='32px system-ui';g.fillText(o.emo,84,10);g.restore();return c}
const RAR=[['LEGENDARY','#ffd36b','\ud83e\udd29'],['RARE','#6aa8ff','\ud83d\ude0e'],['COMMON','#c8c8cc','\ud83d\ude42']];
$('#lookCard').onclick=()=>{if(!hasImg)return;draw();const nz=Object.keys(DEF).filter(k=>S[k]&&Math.abs(S[k])>.02).length,rar=nz>=12?RAR[0]:nz>=7?RAR[1]:RAR[2],a=analyze(cv);let h=0;const ss=state();for(let i=0;i<ss.length;i++)h=(h*31+ss.charCodeAt(i))>>>0;
 const c=cardCanvas({src:cv,w:W,h:H,title:lookName()==='custom'?'My Custom Grade':lookName(),sub:rar[0]+' \u00b7 '+nz+' ADJUSTMENTS',rar,cols:a.cols,stats:[['WARMTH',S.temp],['CONTRAST',S.con],['GRAIN',S.grain],['COLOUR',S.sat]],foot:'LUMINA STUDIO \u00b7 '+new Date().toLocaleDateString()+' \u00b7 No. '+String(h%1000).padStart(3,'0'),emo:rar[2]});tixAdd('card',1,3);c.toBlob(b=>dl(b,fname('_card.png')))};
let HS=null;async function closest(a){if(!window.HERO)return['unknown'];if(!HS)HS=await Promise.all(HERO.map(x=>new Promise(r=>{const im=new Image();im.onload=()=>r(analyze(im));im.src=x[2]})));
 const sc=HS.map((s,i)=>[Math.abs(s.warm-a.warm)*2+Math.abs(s.contrast-a.contrast)+Math.abs(s.sat-a.sat)+Math.abs(s.bright-a.bright),HERO[i][0].split(' \u00b7 ')[0]]).sort((x,y)=>x[0]-y[0]),o=[];for(const[,n]of sc){if(!o.includes(n))o.push(n);if(o.length==2)break}return o}
let fsIn=null;
function frameStudy(){if(!fsIn){fsIn=document.createElement('input');fsIn.type='file';fsIn.accept='image/*';fsIn.hidden=true;document.body.append(fsIn);
 fsIn.onchange=async e=>{const f=e.target.files[0];if(!f)return;const im=await createImageBitmap(f),a=analyze(im),best=await closest(a),rar=a.hues>=40&&a.contrast>.5?RAR[0]:a.hues>=25?RAR[1]:RAR[2];fsIn.value='';
  const c=cardCanvas({src:im,w:im.width,h:im.height,title:'Frame Study',sub:rar[0]+' \u00b7 CLOSEST: '+best.join(' / ').toUpperCase(),rar,cols:a.cols,stats:[['WARMTH',a.warm*3],['CONTRAST',a.contrast],['COLOUR',a.sat*1.6],['BRIGHT',a.bright*2-1]],foot:'LUMINA STUDIO \u00b7 FOR STUDY \u00b7 '+new Date().toLocaleDateString(),emo:rar[2]});
  const old=$('#fsM');if(old)old.remove();const m=mk('div','fsM','<div class="pc"><p class="note">A rough match from tone and colour, not the real grade. For study only.</p><div id="fsc"></div><button class="btn" id="fs1">Save card</button><button class="btn" id="fs2">Match my photo to this frame</button><button class="btn" id="fs3">Close</button></div>');
  $('#fsc').append(c);c.style.cssText='width:100%;max-height:58vh;object-fit:contain;border-radius:10px';$('#fs1').onclick=()=>c.toBlob(b=>dl(b,'frame_study_card.png'));$('#fs2').onclick=()=>{if(!hasImg){toast('Open a photo first');return}$('#palIn').onchange({target:{files:[f]}});m.remove()};$('#fs3').onclick=()=>m.remove()}}fsIn.click()}
const fsB=document.createElement('button');fsB.className='rb';fsB.title='Study a movie frame';fsB.innerHTML='<svg viewBox="0 0 24 24"><rect x="4" y="5" width="16" height="14" rx="1.5"/><path d="M4 9h16M4 15h16M9 5v14M15 5v14"/></svg>';side.append(fsB);fsB.onclick=frameStudy;
// cinema letterbox bars (animated)
const barsB=document.createElement('button');barsB.className='rb';barsB.title='Cinema bars';barsB.innerHTML='<svg viewBox="0 0 24 24"><rect x="3" y="7" width="18" height="10" rx="1"/><path d="M3 4h18M3 20h18"/></svg>';side.append(barsB);
barsB.onclick=()=>{if(!hasImg)return;const ia=W*sq/H,on=!(S.frame>ia),from=S.frame>ia?S.frame:ia,to=on?2.39:ia,t0=performance.now();barsB.classList.toggle('on',on);if(on)S.frame=ia;
 (function f(now){const p=Math.min(1,(now-t0)/700),e=1-Math.pow(1-p,3);S.frame=on||p<1?from+(to-from)*e:0;draw();if(p<1)requestAnimationFrame(f)})(t0)};
// ================= HOME PAGE (simple): four round buttons + now-showing board =================
(function(){const E=$('#empty');if(!E)return;
 const fb=document.createElement('button');fb.className='btn big';fb.id='frameB';$('#liveBtn').after(fb);fb.onclick=frameStudy;
 [['open','<path d="M12 5v14M5 12h14"/>','Open'],['sample','<circle cx="12" cy="12" r="4"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/>','Sample'],['liveBtn','<path d="M4 8h3l2-3h6l2 3h3v11H4z"/><circle cx="12" cy="13" r="3.5"/>','Live'],['frameB','<rect x="4" y="5" width="16" height="14" rx="1.5"/><path d="M4 9h16M4 15h16M9 5v14M15 5v14"/>','Frame']].forEach(([id,p,l])=>{const b=$('#'+id);if(b)b.innerHTML='<svg viewBox="0 0 24 24">'+p+'</svg>'+l});
 const bo=document.createElement('div');bo.id='boxo';($('#mid')||E).append(bo);
 const upd=()=>{bo.textContent='NOW SHOWING \u00b7 '+(MOODNAME||'YOUR STORY').toUpperCase()+' \u00b7 ROLL '+rollN+' \u00b7 '+tixGet()+' TICKETS'};upd();setInterval(upd,2000);
})();
