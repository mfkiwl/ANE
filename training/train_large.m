// train_large.m — Train a single transformer layer FULLY on ANE
// 7 ANE kernels per step:
//   Forward:  kFwdAttn (QKV+SDPA+Wo, taps Q,K,V,attn_out) + kFwdFFN (W1+W3+SiLU+W2, taps h1,h3,silu_out)
//   Backward: kFFNBwd (W2^T+SiLU_bwd+W1^T+W3^T) + kSdpaBwd1 (Wo^T+SDPA) + kSdpaBwd2 + kQKVb (Wq^T+Wk^T+Wv^T)
// CPU: RMSNorm (fwd+bwd), residuals, loss, dW accumulation (cblas), SGD update
// NO CPU recompute of Q,K,V,h1,h3 — all exposed via forward taps
#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#import <mach/mach_time.h>
#import <Accelerate/Accelerate.h>
#include <math.h>
#include <unistd.h>
#include <dispatch/dispatch.h>

#define DIM 768
#define HIDDEN 2048
#define HEADS 12
#define HD (DIM/HEADS)
#define SEQ 512
#define ACCUM_STEPS 100
#define MAX_COMPILES 100
#define NUM_KERNELS 6
#define CKPT_PATH "/tmp/ane_large_ckpt.bin"

static Class g_D, g_I, g_AR, g_AIO;
static mach_timebase_info_data_t g_tb;
static int g_compile_count = 0;

static void ane_init(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_D  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_I  = NSClassFromString(@"_ANEInMemoryModel");
    g_AR = NSClassFromString(@"_ANERequest");
    g_AIO= NSClassFromString(@"_ANEIOSurfaceObject");
}
static double tb_ms(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static IOSurfaceRef make_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth:@(bytes), (id)kIOSurfaceHeight:@1,
        (id)kIOSurfaceBytesPerElement:@1, (id)kIOSurfaceBytesPerRow:@(bytes),
        (id)kIOSurfaceAllocSize:@(bytes), (id)kIOSurfacePixelFormat:@0});
}
static NSData *build_blob(const float *w, int rows, int cols) {
    int ws=rows*cols*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows*cols;i++) fp16[i]=(_Float16)w[i];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_t(const float *w, int rows, int cols) {
    int ws=cols*rows*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    _Float16 *fp16=(_Float16*)(b+128);
    for(int i=0;i<rows;i++) for(int j=0;j<cols;j++) fp16[j*rows+i]=(_Float16)w[i*cols+j];
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}
static NSData *build_blob_fp16(_Float16 *d, int cnt) {
    int ws=cnt*2, tot=128+ws;
    uint8_t *b=(uint8_t*)calloc(tot,1);
    b[0]=1;b[4]=2;b[64]=0xEF;b[65]=0xBE;b[66]=0xAD;b[67]=0xDE;b[68]=1;
    *(uint32_t*)(b+72)=ws;*(uint32_t*)(b+80)=128;
    memcpy(b+128,d,ws);
    return [NSData dataWithBytesNoCopy:b length:tot freeWhenDone:YES];
}

// ===== MIL generators =====

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// SDPA forward + taps: x_in → rmsnorm → QKV+SDPA+Wo → concat(o_out, Q, K, V, attn_out, xnorm) fp16
static NSString *gen_sdpa_fwd_taps(void) {
    float sc = 1.0f/sqrtf((float)HD);
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    // --- RMSNorm: x → xn ---
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", DIM, SEQ];
    // --- QKV + SDPA + Wo (operates on xn) ---
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=q4)[name=string(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=qsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=k4)[name=string(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=qsh,x=vf)[name=string(\"rv\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=v4)[name=string(\"tv\")];\n", HEADS,SEQ,HD];
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ,SEQ,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,qf,kf,vf,af,xn))[name=string(\"cat\")];\n", 6*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// FFN forward + taps: x2 → rmsnorm → FFN → concat(ffn_out, h1, h3, silu_out, x2norm) fp16
static NSString *gen_ffn_fwd_taps(void) {
    float invd = 1.0f/(float)DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM, SEQ];
    // --- RMSNorm: x → xn ---
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendFormat:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", SEQ];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", SEQ];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(0.00001)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", SEQ];
    [m appendFormat:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms2.bin\"), offset=uint64(64)))];\n", DIM, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", DIM, SEQ];
    // --- FFN (operates on xn) ---
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", HIDDEN,DIM,HIDDEN,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", DIM,HIDDEN,DIM,HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> silu = mul(x=h1,y=sig)[name=string(\"si\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gate = mul(x=silu,y=h3)[name=string(\"gt\")];\n", HIDDEN,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> y = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gate)[name=string(\"c2\")];\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(y,h1,h3,gate,xn))[name=string(\"cat\")];\n", 2*DIM+3*HIDDEN,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Fused FFN backward: concat(dffn,h1,h3) → concat(dx,dh1,dh3) fp16
static NSString *gen_ffn_bwd(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@CONV_CONST];
    [m appendString:@"        tensor<int32, [4]> bd = const()[name=string(\"bd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> sd = const()[name=string(\"sd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dffn = slice_by_size(x=x,begin=bd,size=sd)[name=string(\"s0\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<int32, [4]> s1 = const()[name=string(\"s1\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = slice_by_size(x=x,begin=b1,size=s1)[name=string(\"s1x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM+HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = slice_by_size(x=x,begin=b3,size=s1)[name=string(\"s3x\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2t = const()[name=string(\"W2t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2t.bin\"), offset=uint64(64)))];\n", HIDDEN, DIM, HIDDEN, DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsilu = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2t,x=dffn)[name=string(\"cw2\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sig = sigmoid(x=h1)[name=string(\"sg\")];\n", HIDDEN, SEQ];
    [m appendString:@"        fp16 one = const()[name=string(\"one\"), val=fp16(1.0)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oms = sub(x=one,y=sig)[name=string(\"oms\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> homs = mul(x=h1,y=oms)[name=string(\"homs\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> brk = add(x=one,y=homs)[name=string(\"brk\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dsd = mul(x=sig,y=brk)[name=string(\"dsd\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> t1 = mul(x=dsilu,y=h3)[name=string(\"t1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh1 = mul(x=t1,y=dsd)[name=string(\"dh1\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> slh = mul(x=h1,y=sig)[name=string(\"slh\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dh3 = mul(x=dsilu,y=slh)[name=string(\"dh3\")];\n", HIDDEN, SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1t = const()[name=string(\"W1t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1t.bin\"), offset=uint64(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3t = const()[name=string(\"W3t\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3t.bin\"), offset=uint64(64)))];\n", DIM, HIDDEN, DIM, HIDDEN];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1t,x=dh1)[name=string(\"cw1\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3t,x=dh3)[name=string(\"cw3\")];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx = add(x=dx1,y=dx3)[name=string(\"adx\")];\n", DIM, SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dx,dh1,dh3))[name=string(\"cat\")];\n", DIM+2*HIDDEN, SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// Fused QKV backward: concat(dq,dk,dv) → dx fp16
static NSString *gen_qkvb(void) {
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 3*DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dq = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dk = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dv = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wqt = const()[name=string(\"Wqt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wqt.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wkt = const()[name=string(\"Wkt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wkt.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wvt = const()[name=string(\"Wvt\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wvt.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxq = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wqt,x=dq)[name=string(\"cq\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxk = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wkt,x=dk)[name=string(\"ck\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxv = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wvt,x=dv)[name=string(\"cv\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dxqk = add(x=dxq,y=dxk)[name=string(\"aqk\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=dxqk,y=dxv)[name=string(\"out\")];\n", DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 1 + Wo^T: concat(Q,K,V,dx2) → Wo^T(dx2) → concat(dV, probs_flat, dp_flat) fp16
// SCORE_CH: channels needed for flattened attention scores [HEADS,SEQ,SEQ] → [HEADS*SEQ, SEQ]
#define SCORE_CH (HEADS*SEQ)

static NSString *gen_sdpa_bwd1(void) {
    float sc = 1.0f/sqrtf((float)HD);
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", 4*DIM, SEQ];
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<int32, [4]> sz = const()[name=string(\"sz\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b0,size=sz)[name=string(\"s0\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b1,size=sz)[name=string(\"s1\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = slice_by_size(x=x,begin=b2,size=sz)[name=string(\"s2\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 3*DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dx2f = slice_by_size(x=x,begin=b3,size=sz)[name=string(\"s3\")];\n", DIM,SEQ];
    // Wo^T backward: dx2 → dattn
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wot = const()[name=string(\"Wot\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wot.bin\"), offset=uint64(64)))];\n", DIM,DIM,DIM,DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> df = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wot,x=dx2f)[name=string(\"cwo\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> vr = reshape(shape=rsh,x=vf)[name=string(\"rv\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=vr)[name=string(\"tv\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dr = reshape(shape=rsh,x=df)[name=string(\"rd\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> da = transpose(perm=pm,x=dr)[name=string(\"td\")];\n", HEADS,SEQ,HD];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=bF,transpose_y=bT,x=q,y=k)[name=string(\"mm1\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", SEQ,SEQ,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dv4 = matmul(transpose_x=bT,transpose_y=bF,x=probs,y=da)[name=string(\"dv\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp4 = matmul(transpose_x=bF,transpose_y=bT,x=da,y=v)[name=string(\"dp\")];\n", HEADS,SEQ,SEQ];
    // Flatten dv back to [1,DIM,1,SEQ]
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dvt = transpose(perm=pm,x=dv4)[name=string(\"dvt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> dvs = const()[name=string(\"dvs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dvf = reshape(shape=dvs,x=dvt)[name=string(\"dvf\")];\n", DIM,SEQ];
    // Flatten probs [1,H,S,S] → [1,H*S,1,S] and dp [1,H,S,S] → [1,H*S,1,S]
    [m appendFormat:@"        tensor<int32, [4]> scs = const()[name=string(\"scs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = reshape(shape=scs,x=probs)[name=string(\"pf\")];\n", SCORE_CH,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = reshape(shape=scs,x=dp4)[name=string(\"dpf\")];\n", SCORE_CH,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dvf,pf,dpf))[name=string(\"cat\")];\n", DIM+2*SCORE_CH,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// SDPA backward part 2: concat(probs[SCORE_CH],dp[SCORE_CH],Q[DIM],K[DIM]) → concat(dQ,dK) fp16
static NSString *gen_sdpa_bwd2(void) {
    float sc = 1.0f/sqrtf((float)HD);
    int bwd2_in = 2*SCORE_CH + 2*DIM;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", bwd2_in, SEQ];
    // Slice probs
    [m appendFormat:@"        tensor<int32, [4]> sz_sc = const()[name=string(\"szsc\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", SCORE_CH, SEQ];
    [m appendString:@"        tensor<int32, [4]> b0 = const()[name=string(\"b0\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> pf = slice_by_size(x=x,begin=b0,size=sz_sc)[name=string(\"s0\")];\n", SCORE_CH,SEQ];
    // Slice dp
    [m appendFormat:@"        tensor<int32, [4]> b1 = const()[name=string(\"b1\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dpf = slice_by_size(x=x,begin=b1,size=sz_sc)[name=string(\"s1\")];\n", SCORE_CH,SEQ];
    // Slice Q
    [m appendFormat:@"        tensor<int32, [4]> sz_d = const()[name=string(\"szd\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM, SEQ];
    [m appendFormat:@"        tensor<int32, [4]> b2 = const()[name=string(\"b2\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = slice_by_size(x=x,begin=b2,size=sz_d)[name=string(\"s2\")];\n", DIM,SEQ];
    // Slice K
    [m appendFormat:@"        tensor<int32, [4]> b3 = const()[name=string(\"b3\"), val=tensor<int32, [4]>([0,%d,0,0])];\n", 2*SCORE_CH+DIM];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = slice_by_size(x=x,begin=b3,size=sz_d)[name=string(\"s3\")];\n", DIM,SEQ];
    // Reshape to multi-head
    [m appendFormat:@"        tensor<int32, [4]> ssh = const()[name=string(\"ssh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> probs = reshape(shape=ssh,x=pf)[name=string(\"rp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dp = reshape(shape=ssh,x=dpf)[name=string(\"rdp\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> rsh = const()[name=string(\"rsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", HEADS,HD,SEQ];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr = reshape(shape=rsh,x=qf)[name=string(\"rq\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=qr)[name=string(\"tq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr = reshape(shape=rsh,x=kf)[name=string(\"rk\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=kr)[name=string(\"tk\")];\n", HEADS,SEQ,HD];
    // Softmax grad: ds = probs * (dp - sum(probs*dp)) * scale
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> pdp = mul(x=probs,y=dp)[name=string(\"pdp\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([-1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,1]> spdp = reduce_sum(x=pdp,axes=rax,keep_dims=kd)[name=string(\"rs\")];\n", HEADS,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dps = sub(x=dp,y=spdp)[name=string(\"dps\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds0 = mul(x=probs,y=dps)[name=string(\"ds0\")];\n", HEADS,SEQ,SEQ];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", sc];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ds = mul(x=ds0,y=scv)[name=string(\"ds\")];\n", HEADS,SEQ,SEQ];
    [m appendString:@"        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n"];
    [m appendString:@"        bool bT = const()[name=string(\"bT\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dq4 = matmul(transpose_x=bF,transpose_y=bF,x=ds,y=k)[name=string(\"dq\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dk4 = matmul(transpose_x=bT,transpose_y=bF,x=ds,y=q)[name=string(\"dk\")];\n", HEADS,SEQ,HD];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dqt = transpose(perm=pm,x=dq4)[name=string(\"dqt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> dkt = transpose(perm=pm,x=dk4)[name=string(\"dkt\")];\n", HEADS,HD,SEQ];
    [m appendFormat:@"        tensor<int32, [4]> fs = const()[name=string(\"fs\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dqf = reshape(shape=fs,x=dqt)[name=string(\"dqf\")];\n", DIM,SEQ];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> dkf = reshape(shape=fs,x=dkt)[name=string(\"dkf\")];\n", DIM,SEQ];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(dqf,dkf))[name=string(\"cat\")];\n", 2*DIM,SEQ];
    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// ===== Weight builders =====
static NSData *g_mask_blob = nil;
static NSData *get_mask_blob(void) {
    if (!g_mask_blob) {
        _Float16 *mask = (_Float16*)calloc(SEQ*SEQ, sizeof(_Float16));
        for(int t=0;t<SEQ;t++) for(int t2=0;t2<SEQ;t2++)
            mask[t*SEQ+t2] = (t2<=t) ? (_Float16)0.0f : (_Float16)(-65504.0f);
        g_mask_blob = build_blob_fp16(mask, SEQ*SEQ);
        free(mask);
    }
    return g_mask_blob;
}

// ===== Kernel compilation and evaluation =====
typedef struct { void *model; IOSurfaceRef ioIn, ioOut; void *request; void *tmpDir; } Kern;

static Kern *compile_kern_mil_w(NSString *mil, NSDictionary *weights, int ic_bytes, int oc_bytes) {
    @autoreleasepool {
    NSData *md = [mil dataUsingEncoding:NSUTF8StringEncoding];
    id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(g_D, @selector(modelWithMILText:weights:optionsPlist:), md, weights, nil);
    if (!desc) { printf("  [compile] desc=NULL\n"); return NULL; }
    id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(g_I, @selector(inMemoryModelWithDescriptor:), desc);
    id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
    NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
    [[NSFileManager defaultManager] createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"] withIntermediateDirectories:YES attributes:nil error:nil];
    [md writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
    for (NSString *path in weights) {
        NSString *rel = [path stringByReplacingOccurrencesOfString:@"@model_path/" withString:@""];
        [weights[path][@"data"] writeToFile:[td stringByAppendingPathComponent:rel] atomically:YES];
    }
    NSError *e = nil;
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] FAIL: %s\n", e ? [[e description] UTF8String] : "no error"); return NULL;
    }
    if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e)) {
        printf("  [compile] load FAIL\n"); return NULL;
    }
    __sync_fetch_and_add(&g_compile_count, 1);
    Kern *k = calloc(1, sizeof(Kern));
    k->model = CFBridgingRetain(mdl);
    k->ioIn = make_surface(ic_bytes);
    k->ioOut = make_surface(oc_bytes);
    id wI = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioIn);
    id wO = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(g_AIO, @selector(objectWithIOSurface:), k->ioOut);
    k->request = CFBridgingRetain(((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(g_AR,
        @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
        @[wI], @[@0], @[wO], @[@0], nil, nil, @0));
    k->tmpDir = CFBridgingRetain(td);
    return k;
    }
}
static void free_kern(Kern *k) {
    if (!k) return;
    id mdl = (__bridge id)k->model; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(mdl, @selector(unloadWithQoS:error:), 21, &e);
    CFRelease(k->ioIn); CFRelease(k->ioOut);
    [[NSFileManager defaultManager] removeItemAtPath:(__bridge id)k->tmpDir error:nil];
    CFRelease(k->model); CFRelease(k->request); CFRelease(k->tmpDir);
    free(k);
}
static void ane_eval(Kern *k) {
    id mdl = (__bridge id)k->model; id req = (__bridge id)k->request; NSError *e = nil;
    ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(mdl, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
}

// ===== Vectorized conversion helpers (NEON) =====
#include <arm_neon.h>
static void cvt_f16_f32(float *dst, const _Float16 *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vld1q_f16((const __fp16*)(src+i));
        vst1q_f32(dst+i,   vcvt_f32_f16(vget_low_f16(h)));
        vst1q_f32(dst+i+4, vcvt_f32_f16(vget_high_f16(h)));
    }
    for (; i < n; i++) dst[i] = (float)src[i];
}
static void cvt_f32_f16(_Float16 *dst, const float *src, int n) {
    int i = 0;
    for (; i+7 < n; i += 8) {
        float16x8_t h = vcombine_f16(vcvt_f16_f32(vld1q_f32(src+i)),
                                      vcvt_f16_f32(vld1q_f32(src+i+4)));
        vst1q_f16((__fp16*)(dst+i), h);
    }
    for (; i < n; i++) dst[i] = (_Float16)src[i];
}

// ===== IOSurface I/O helpers (channel-first, no transpose) =====
// All CPU buffers are [C,S] channel-first matching IOSurface [1,C,1,S]
// Write fp32 [C,S] → fp16 [1,C,1,S] (just type conversion, no transpose)
static void io_write_fp16(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *dst = (_Float16*)IOSurfaceGetBaseAddress(s);
    cvt_f32_f16(dst, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}
// Write fp32 [C,S] → fp32 [1,C,1,S] (just memcpy)
static void io_write_fp32(IOSurfaceRef s, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(s), data, channels * sp * sizeof(float));
    IOSurfaceUnlock(s, 0, NULL);
}
// Read fp16 [1,C,1,S] → fp32 [C,S] at channel offset (just type conversion)
static void io_read_fp16(IOSurfaceRef s, float *data, int ch_off, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    _Float16 *src = (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp;
    cvt_f16_f32(data, src, channels * sp);
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
// Read fp32 [1,C,1,S] → fp32 [C,S] (just memcpy)
static void io_read_fp32(IOSurfaceRef s, float *data, int channels, int sp) {
    IOSurfaceLock(s, kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(s), channels * sp * sizeof(float));
    IOSurfaceUnlock(s, kIOSurfaceLockReadOnly, NULL);
}
// Write multiple fp32 [C,S] arrays concatenated along channel dim as fp16
static void io_write_multi_fp16(IOSurfaceRef s, int sp, int n, ...) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *dst = (_Float16*)IOSurfaceGetBaseAddress(s);
    va_list ap; va_start(ap, n);
    int ch_off = 0;
    for (int i=0; i<n; i++) {
        const float *data = va_arg(ap, const float*);
        int channels = va_arg(ap, int);
        cvt_f32_f16(dst + ch_off*sp, data, channels * sp);
        ch_off += channels;
    }
    va_end(ap);
    IOSurfaceUnlock(s, 0, NULL);
}
// Direct copy between IOSurfaces (no format conversion — both fp16 channel-first)
static void io_copy(IOSurfaceRef dst, int dst_ch, IOSurfaceRef src, int src_ch, int channels, int sp) {
    IOSurfaceLock(dst, 0, NULL);
    IOSurfaceLock(src, kIOSurfaceLockReadOnly, NULL);
    memcpy((_Float16*)IOSurfaceGetBaseAddress(dst) + dst_ch*sp,
           (_Float16*)IOSurfaceGetBaseAddress(src) + src_ch*sp,
           channels * sp * sizeof(_Float16));
    IOSurfaceUnlock(src, kIOSurfaceLockReadOnly, NULL);
    IOSurfaceUnlock(dst, 0, NULL);
}
// Write one fp32 [C,S] array at specific channel offset in IOSurface as fp16
static void io_write_fp16_at(IOSurfaceRef s, int ch_off, const float *data, int channels, int sp) {
    IOSurfaceLock(s, 0, NULL);
    _Float16 *dst = (_Float16*)IOSurfaceGetBaseAddress(s) + ch_off * sp;
    cvt_f32_f16(dst, data, channels * sp);
    IOSurfaceUnlock(s, 0, NULL);
}

// ===== CPU ops (channel-first [C,S] layout) =====
// x[i*S+t] = channel i, position t
// Process all positions in parallel using vectorized column ops
static float *g_rms_tmp = NULL;
static void rmsnorm(float *out, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = malloc(S*4);
    float *ss = calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    int n = S; vvrsqrtf(ss, ss, &n);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, ss, 1, out+i*S, 1, (vDSP_Length)S);
        vDSP_vsmul(out+i*S, 1, &w[i], out+i*S, 1, (vDSP_Length)S);
    }
    free(ss);
}
static void rmsnorm_bwd(float *dx, float *dw, const float *dy, const float *x, const float *w, int d, int S) {
    if (!g_rms_tmp) g_rms_tmp = malloc(S*4);
    float *ss = calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vadd(g_rms_tmp, 1, ss, 1, ss, 1, (vDSP_Length)S);
    }
    float invd = 1.0f/d, eps=1e-5f;
    vDSP_vsmsa(ss, 1, &invd, &eps, ss, 1, (vDSP_Length)S);
    float *rrms = malloc(S*4);
    int n = S; vvrsqrtf(rrms, ss, &n);
    // dot[t] = sum_i dy[i,t]*x[i,t]*w[i]
    float *dot = calloc(S, sizeof(float));
    for (int i=0; i<d; i++) {
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        // dot += tmp * w[i]  (vDSP_vsma: A*scalar+C→D)
        vDSP_vsma(g_rms_tmp, 1, &w[i], dot, 1, dot, 1, (vDSP_Length)S);
    }
    // dot *= rrms^2/d
    vDSP_vmul(rrms, 1, rrms, 1, ss, 1, (vDSP_Length)S);
    vDSP_vsmul(ss, 1, &invd, ss, 1, (vDSP_Length)S);
    vDSP_vmul(dot, 1, ss, 1, dot, 1, (vDSP_Length)S);
    for (int i=0; i<d; i++) {
        vDSP_vmul(x+i*S, 1, dot, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsub(g_rms_tmp, 1, dy+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vsmul(g_rms_tmp, 1, &w[i], dx+i*S, 1, (vDSP_Length)S);
        // dw[i] += sum_t dy[i,t]*x[i,t]*rrms[t]
        vDSP_vmul(dy+i*S, 1, x+i*S, 1, g_rms_tmp, 1, (vDSP_Length)S);
        vDSP_vmul(g_rms_tmp, 1, rrms, 1, g_rms_tmp, 1, (vDSP_Length)S);
        float s; vDSP_sve(g_rms_tmp, 1, &s, (vDSP_Length)S);
        dw[i] += s;
    }
    free(ss); free(rrms); free(dot);
}

// ===== Checkpoint =====
typedef struct {
    int step, total_steps;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches;
    int adam_t; // Adam timestep
} CkptHdr;

// Adam optimizer state
typedef struct {
    float *m, *v; // first and second moment
    size_t n;
} AdamState;
static AdamState adam_alloc(size_t n) { return (AdamState){calloc(n,4), calloc(n,4), n}; }
static void adam_free(AdamState *s) { free(s->m); free(s->v); }
static void adam_update(float *w, const float *g, AdamState *s, int t, float lr, float b1, float b2, float eps) {
    float bc1 = 1.0f - powf(b1, t), bc2 = 1.0f - powf(b2, t);
    for (size_t i=0; i<s->n; i++) {
        s->m[i] = b1*s->m[i] + (1-b1)*g[i];
        s->v[i] = b2*s->v[i] + (1-b2)*g[i]*g[i];
        float mh = s->m[i]/bc1, vh = s->v[i]/bc2;
        w[i] -= lr * mh / (sqrtf(vh) + eps);
    }
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        ane_init();
        mach_timebase_info(&g_tb);

        int total_steps = 400;
        float lr = 1e-3f;
        float adam_b1=0.9f, adam_b2=0.999f, adam_eps=1e-8f;
        int adam_t = 0;
        int start_step = 0;

        size_t wq_sz = DIM*DIM, wo_sz = DIM*DIM;
        size_t w1_sz = HIDDEN*DIM, w2_sz = DIM*HIDDEN, w3_sz = HIDDEN*DIM;
        size_t total_params = 4*wq_sz + w1_sz + w2_sz + w3_sz;

        float *Wq=malloc(wq_sz*4), *Wk=malloc(wq_sz*4), *Wv=malloc(wq_sz*4), *Wo=malloc(wo_sz*4);
        float *W1=malloc(w1_sz*4), *W2=malloc(w2_sz*4), *W3=malloc(w3_sz*4);
        float *rms1_w=malloc(DIM*4), *rms2_w=malloc(DIM*4);

        // Adam optimizer states (m and v for each weight)
        AdamState aWq=adam_alloc(wq_sz), aWk=adam_alloc(wq_sz), aWv=adam_alloc(wq_sz), aWo=adam_alloc(wo_sz);
        AdamState aW1=adam_alloc(w1_sz), aW2=adam_alloc(w2_sz), aW3=adam_alloc(w3_sz);
        AdamState arms1=adam_alloc(DIM), arms2=adam_alloc(DIM);

        double cum_compile=0, cum_train=0, cum_wall=0;
        int cum_steps=0, cum_batches=0;

        bool resuming = false;
        if (argc > 1 && strcmp(argv[1], "--resume") == 0) {
            FILE *f = fopen(CKPT_PATH, "rb");
            if (f) {
                CkptHdr h; fread(&h, sizeof(h), 1, f);
                start_step=h.step; total_steps=h.total_steps; lr=h.lr;
                cum_compile=h.cum_compile; cum_train=h.cum_train; cum_wall=h.cum_wall;
                cum_steps=h.cum_steps; cum_batches=h.cum_batches; adam_t=h.adam_t;
                fread(Wq,4,wq_sz,f); fread(Wk,4,wq_sz,f); fread(Wv,4,wq_sz,f); fread(Wo,4,wo_sz,f);
                fread(W1,4,w1_sz,f); fread(W2,4,w2_sz,f); fread(W3,4,w3_sz,f);
                fread(rms1_w,4,DIM,f); fread(rms2_w,4,DIM,f);
                // Adam state
                fread(aWq.m,4,wq_sz,f);fread(aWq.v,4,wq_sz,f);
                fread(aWk.m,4,wq_sz,f);fread(aWk.v,4,wq_sz,f);
                fread(aWv.m,4,wq_sz,f);fread(aWv.v,4,wq_sz,f);
                fread(aWo.m,4,wo_sz,f);fread(aWo.v,4,wo_sz,f);
                fread(aW1.m,4,w1_sz,f);fread(aW1.v,4,w1_sz,f);
                fread(aW2.m,4,w2_sz,f);fread(aW2.v,4,w2_sz,f);
                fread(aW3.m,4,w3_sz,f);fread(aW3.v,4,w3_sz,f);
                fread(arms1.m,4,DIM,f);fread(arms1.v,4,DIM,f);
                fread(arms2.m,4,DIM,f);fread(arms2.v,4,DIM,f);
                fclose(f);
                resuming = true;
                printf("[RESUMED step %d, loss=%.6f]\n", start_step, h.loss);
            }
        }
        if (!resuming) {
            srand48(42);
            float scale_d=1.0f/sqrtf(DIM), scale_h=1.0f/sqrtf(HIDDEN);
            for(size_t i=0;i<wq_sz;i++){Wq[i]=scale_d*(2*drand48()-1);Wk[i]=scale_d*(2*drand48()-1);}
            for(size_t i=0;i<wq_sz;i++){Wv[i]=scale_d*(2*drand48()-1);Wo[i]=scale_d*(2*drand48()-1);}
            for(size_t i=0;i<w1_sz;i++) W1[i]=scale_h*(2*drand48()-1);
            for(size_t i=0;i<w2_sz;i++) W2[i]=scale_d*(2*drand48()-1);
            for(size_t i=0;i<w3_sz;i++) W3[i]=scale_h*(2*drand48()-1);
            for(int i=0;i<DIM;i++){rms1_w[i]=1.0f; rms2_w[i]=1.0f;}
        }

        if (!resuming) {
            // FLOP accounting: 7 weight matrices, each 2*OC*IC*SEQ for forward
            double fwd_flops = 4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ;
            double bwd_dx_flops = fwd_flops; // same matmuls transposed
            double bwd_dw_flops = fwd_flops; // dW = dy^T @ x, same FLOPs
            double sdpa_flops = 2.0*HEADS*5*SEQ*SEQ*HD; // 5 SEQ×SEQ matmuls in backward
            double total_flops = fwd_flops + bwd_dx_flops + bwd_dw_flops + sdpa_flops;
            double ane_flops_step = fwd_flops + bwd_dx_flops + sdpa_flops;
            printf("=== ANE Training: Fully-ANE Pipeline ===\n");
            printf("dim=%d hidden=%d heads=%d seq=%d\n", DIM, HIDDEN, HEADS, SEQ);
            printf("Params: %.2fM | Weights: %.1fMB FP16\n", total_params/1e6, total_params*2.0/1e6);
            printf("Kernels: %d (fwdAttn+fwdFFN+ffnBwd+sdpaBwd1+sdpaBwd2+qkvBwd)\n", NUM_KERNELS);
            printf("Accum %d steps per recompile | Adam LR=%.1e b1=%.1f b2=%.3f\n\n", ACCUM_STEPS, lr, adam_b1, adam_b2);
            printf("FLOPs/step: fwd=%.0fM bwd_dx=%.0fM bwd_dW=%.0fM sdpa_bwd=%.0fM total=%.0fM\n",
                   fwd_flops/1e6, bwd_dx_flops/1e6, bwd_dw_flops/1e6, sdpa_flops/1e6, total_flops/1e6);
            printf("ANE FLOPs/step: %.0fM (fwd+bwd_dx+sdpa_bwd) | CPU: dW (cblas)\n\n", ane_flops_step/1e6);
        }

        // Training data
        float *x_in=malloc(SEQ*DIM*4), *y_tgt=malloc(SEQ*DIM*4);
        // Training data in channel-first [C,S] layout
        if (!resuming) srand48(42);
        for(int c=0;c<DIM;c++) for(int t=0;t<SEQ;t++) {
            int idx = c*SEQ+t;
            x_in[idx]=0.1f*(2*drand48()-1);
            y_tgt[idx]=0.1f*sinf(idx*0.03f+1.0f);
        }

        // Activation buffers (saved from forward for backward)
        float *xnorm=malloc(SEQ*DIM*4);
        float *Q=malloc(SEQ*DIM*4), *K=malloc(SEQ*DIM*4), *V=malloc(SEQ*DIM*4);
        float *attn_out=malloc(SEQ*DIM*4), *o_out=malloc(SEQ*DIM*4);
        float *x2=malloc(SEQ*DIM*4), *x2norm=malloc(SEQ*DIM*4);
        float *h1=malloc(SEQ*HIDDEN*4), *h3=malloc(SEQ*HIDDEN*4), *silu_out=malloc(SEQ*HIDDEN*4);
        float *ffn_out=malloc(SEQ*DIM*4), *y_out=malloc(SEQ*DIM*4);

        // Gradient buffers
        float *dy=malloc(SEQ*DIM*4), *dffn=malloc(SEQ*DIM*4);
        float *dh1=malloc(SEQ*HIDDEN*4), *dh3=malloc(SEQ*HIDDEN*4);
        float *dx_ffn=malloc(SEQ*DIM*4), *dx2=malloc(SEQ*DIM*4);
        float *do_out_buf=malloc(SEQ*DIM*4), *dattn=malloc(SEQ*DIM*4);
        float *dq=malloc(SEQ*DIM*4), *dk=malloc(SEQ*DIM*4), *dv=malloc(SEQ*DIM*4);
        float *dx_attn=malloc(SEQ*DIM*4);
        // SDPA bwd intermediates
        float *probs_flat=malloc(SEQ*DIM*4), *dp_flat=malloc(SEQ*DIM*4);

        // Gradient accumulators
        float *gWq=calloc(wq_sz,4), *gWk=calloc(wq_sz,4), *gWv=calloc(wq_sz,4), *gWo=calloc(wo_sz,4);
        float *gW1=calloc(w1_sz,4), *gW2=calloc(w2_sz,4), *gW3=calloc(w3_sz,4);
        float *grms1=calloc(DIM,4), *grms2=calloc(DIM,4);

        // 7 ANE kernels
        Kern *kFwdAttn=NULL, *kFwdFFN=NULL, *kFFNBwd=NULL;
        Kern *kSdpaBwd1=NULL, *kSdpaBwd2=NULL, *kQKVb=NULL;

        // Compile static (weight-free) kernels ONCE
        kSdpaBwd2 = compile_kern_mil_w(gen_sdpa_bwd2(), @{},
            (2*SCORE_CH+2*DIM)*SEQ*2, 2*DIM*SEQ*2);
        if (!kSdpaBwd2) { printf("Static kernel compile failed\n"); return 1; }

        // GCD queue for async dW cblas (overlaps with ANE evals)
        dispatch_queue_t dw_q = dispatch_queue_create("dw_cblas", DISPATCH_QUEUE_SERIAL);
        dispatch_group_t dw_grp = dispatch_group_create();

        float last_loss = 999.0f;
        double total_compile_ms=0, total_train_ms=0;
        int total_steps_done=0, total_batches=0;
        uint64_t t_wall_start = mach_absolute_time();

        int step = start_step;
        while (step < total_steps) {
            // Check compile budget — 5 weight-bearing kernels per batch
            if (g_compile_count + 5 > MAX_COMPILES) {
                free_kern(kFwdAttn);free_kern(kFwdFFN);free_kern(kFFNBwd);
                free_kern(kSdpaBwd1);free_kern(kQKVb);
                free_kern(kSdpaBwd2);
                double wall = tb_ms(mach_absolute_time() - t_wall_start);
                FILE *f = fopen(CKPT_PATH, "wb");
                CkptHdr h = {step,total_steps,lr,last_loss,
                    total_compile_ms+cum_compile, total_train_ms+cum_train, wall+cum_wall,
                    total_steps_done+cum_steps, total_batches+cum_batches, adam_t};
                fwrite(&h,sizeof(h),1,f);
                fwrite(Wq,4,wq_sz,f);fwrite(Wk,4,wq_sz,f);fwrite(Wv,4,wq_sz,f);fwrite(Wo,4,wo_sz,f);
                fwrite(W1,4,w1_sz,f);fwrite(W2,4,w2_sz,f);fwrite(W3,4,w3_sz,f);
                fwrite(rms1_w,4,DIM,f);fwrite(rms2_w,4,DIM,f);
                // Adam state
                fwrite(aWq.m,4,wq_sz,f);fwrite(aWq.v,4,wq_sz,f);
                fwrite(aWk.m,4,wq_sz,f);fwrite(aWk.v,4,wq_sz,f);
                fwrite(aWv.m,4,wq_sz,f);fwrite(aWv.v,4,wq_sz,f);
                fwrite(aWo.m,4,wo_sz,f);fwrite(aWo.v,4,wo_sz,f);
                fwrite(aW1.m,4,w1_sz,f);fwrite(aW1.v,4,w1_sz,f);
                fwrite(aW2.m,4,w2_sz,f);fwrite(aW2.v,4,w2_sz,f);
                fwrite(aW3.m,4,w3_sz,f);fwrite(aW3.v,4,w3_sz,f);
                fwrite(arms1.m,4,DIM,f);fwrite(arms1.v,4,DIM,f);
                fwrite(arms2.m,4,DIM,f);fwrite(arms2.v,4,DIM,f);
                fclose(f);
                printf("[exec() restart step %d, %d compiles, loss=%.6f]\n", step, g_compile_count, last_loss);
                fflush(stdout);
                execl(argv[0], argv[0], "--resume", NULL);
                perror("execl"); return 1;
            }

            // Compile 5 weight-bearing kernels (sdpaBwd2 compiled once above)
            uint64_t tc = mach_absolute_time();
            free_kern(kFwdAttn);free_kern(kFwdFFN);free_kern(kFFNBwd);
            free_kern(kSdpaBwd1);free_kern(kQKVb);

            kFwdAttn = compile_kern_mil_w(gen_sdpa_fwd_taps(), (@{
                @"@model_path/weights/rms1.bin": @{@"offset":@0, @"data":build_blob(rms1_w,1,DIM)},
                @"@model_path/weights/wq.bin": @{@"offset":@0, @"data":build_blob(Wq,DIM,DIM)},
                @"@model_path/weights/wk.bin": @{@"offset":@0, @"data":build_blob(Wk,DIM,DIM)},
                @"@model_path/weights/wv.bin": @{@"offset":@0, @"data":build_blob(Wv,DIM,DIM)},
                @"@model_path/weights/wo.bin": @{@"offset":@0, @"data":build_blob(Wo,DIM,DIM)},
                @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
            }), DIM*SEQ*2, 6*DIM*SEQ*2);

            kFwdFFN = compile_kern_mil_w(gen_ffn_fwd_taps(), (@{
                @"@model_path/weights/rms2.bin": @{@"offset":@0, @"data":build_blob(rms2_w,1,DIM)},
                @"@model_path/weights/w1.bin": @{@"offset":@0, @"data":build_blob(W1,HIDDEN,DIM)},
                @"@model_path/weights/w3.bin": @{@"offset":@0, @"data":build_blob(W3,HIDDEN,DIM)},
                @"@model_path/weights/w2.bin": @{@"offset":@0, @"data":build_blob(W2,DIM,HIDDEN)},
            }), DIM*SEQ*2, (2*DIM+3*HIDDEN)*SEQ*2);

            kFFNBwd = compile_kern_mil_w(gen_ffn_bwd(), (@{
                @"@model_path/weights/w2t.bin": @{@"offset":@0, @"data":build_blob_t(W2,DIM,HIDDEN)},
                @"@model_path/weights/w1t.bin": @{@"offset":@0, @"data":build_blob_t(W1,HIDDEN,DIM)},
                @"@model_path/weights/w3t.bin": @{@"offset":@0, @"data":build_blob_t(W3,HIDDEN,DIM)},
            }), (DIM+2*HIDDEN)*SEQ*2, (DIM+2*HIDDEN)*SEQ*2);

            kSdpaBwd1 = compile_kern_mil_w(gen_sdpa_bwd1(), (@{
                @"@model_path/weights/mask.bin": @{@"offset":@0, @"data":get_mask_blob()},
                @"@model_path/weights/wot.bin": @{@"offset":@0, @"data":build_blob_t(Wo,DIM,DIM)},
            }), 4*DIM*SEQ*2, (DIM+2*SCORE_CH)*SEQ*2);

            kQKVb = compile_kern_mil_w(gen_qkvb(), (@{
                @"@model_path/weights/wqt.bin": @{@"offset":@0, @"data":build_blob_t(Wq,DIM,DIM)},
                @"@model_path/weights/wkt.bin": @{@"offset":@0, @"data":build_blob_t(Wk,DIM,DIM)},
                @"@model_path/weights/wvt.bin": @{@"offset":@0, @"data":build_blob_t(Wv,DIM,DIM)},
            }), 3*DIM*SEQ*2, DIM*SEQ*2);

            double cms = tb_ms(mach_absolute_time() - tc);
            total_compile_ms += cms;
            if (!kFwdAttn||!kFwdFFN||!kFFNBwd||!kSdpaBwd1||!kQKVb) {
                printf("Compile failed at step %d, restart\n", step);
                g_compile_count = MAX_COMPILES; continue;
            }

            // === Training loop ===
            memset(gWq,0,wq_sz*4);memset(gWk,0,wq_sz*4);memset(gWv,0,wq_sz*4);memset(gWo,0,wo_sz*4);
            memset(gW1,0,w1_sz*4);memset(gW2,0,w2_sz*4);memset(gW3,0,w3_sz*4);
            memset(grms1,0,DIM*4);memset(grms2,0,DIM*4);
            int steps_batch = 0;
            uint64_t tt = mach_absolute_time();

            double t_ane=0,t_io=0,t_elem=0,t_rms=0,t_cblas_wait=0;
            for (int a=0; a<ACCUM_STEPS && step<total_steps; a++, step++) {
                uint64_t t0,t1;
                // ===== FORWARD =====
                // Attention fwd (ANE does rmsnorm internally): x_in → o_out,Q,K,V,attn_out,xnorm
                t0=mach_absolute_time();
                io_write_fp16(kFwdAttn->ioIn, x_in, DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(kFwdAttn);
                t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                // Wait for prev step's dW cblas before reading attn_out/xnorm
                dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
                t1=mach_absolute_time(); t_cblas_wait+=tb_ms(t1-t0); t0=t1;
                io_read_fp16(kFwdAttn->ioOut, o_out,    0,     DIM, SEQ);
                io_read_fp16(kFwdAttn->ioOut, attn_out, 4*DIM, DIM, SEQ);
                io_read_fp16(kFwdAttn->ioOut, xnorm,    5*DIM, DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                for(int i=0;i<SEQ*DIM;i++) x2[i] = x_in[i] + o_out[i];
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                // FFN fwd (ANE does rmsnorm internally): x2 → ffn_out,h1,h3,silu_out,x2norm
                io_write_fp16(kFwdFFN->ioIn, x2, DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(kFwdFFN);
                t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                io_read_fp16(kFwdFFN->ioOut, ffn_out,  0,              DIM,    SEQ);
                io_read_fp16(kFwdFFN->ioOut, h1,       DIM,            HIDDEN, SEQ);
                io_read_fp16(kFwdFFN->ioOut, h3,       DIM+HIDDEN,     HIDDEN, SEQ);
                io_read_fp16(kFwdFFN->ioOut, silu_out, DIM+2*HIDDEN,   HIDDEN, SEQ);
                io_read_fp16(kFwdFFN->ioOut, x2norm,   DIM+3*HIDDEN,   DIM,    SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                // Residual + Loss
                for(int i=0;i<SEQ*DIM;i++) y_out[i] = x2[i] + ffn_out[i];
                float loss = 0;
                for(int i=0;i<SEQ*DIM;i++){
                    float d = y_out[i]-y_tgt[i]; loss += d*d;
                    dy[i] = 2.0f*d/(SEQ*DIM);
                }
                loss /= (SEQ*DIM);
                last_loss = loss;
                memcpy(dffn, dy, SEQ*DIM*4);
                t1=mach_absolute_time(); t_elem+=tb_ms(t1-t0); t0=t1;

                // ===== BACKWARD =====
                // FFN backward (ANE)
                io_write_fp16_at(kFFNBwd->ioIn, 0, dffn, DIM, SEQ);
                io_copy(kFFNBwd->ioIn, DIM, kFwdFFN->ioOut, DIM, 2*HIDDEN, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(kFFNBwd);
                t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                io_read_fp16(kFFNBwd->ioOut, dx_ffn, 0,           DIM,    SEQ);
                io_read_fp16(kFFNBwd->ioOut, dh1,    DIM,         HIDDEN, SEQ);
                io_read_fp16(kFFNBwd->ioOut, dh3,    DIM+HIDDEN,  HIDDEN, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                // dW FFN async (overlaps with rmsnorm2_bwd + SDPA)
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, HIDDEN, SEQ,
                                1.0f, dffn, SEQ, silu_out, SEQ, 1.0f, gW2, HIDDEN);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, dh1, SEQ, x2norm, SEQ, 1.0f, gW1, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, HIDDEN, DIM, SEQ,
                                1.0f, dh3, SEQ, x2norm, SEQ, 1.0f, gW3, DIM);
                });

                // RMSNorm2 backward — runs in parallel with dW FFN
                memset(dx2, 0, SEQ*DIM*4);
                rmsnorm_bwd(dx2, grms2, dx_ffn, x2, rms2_w, DIM, SEQ);
                for(int i=0;i<SEQ*DIM;i++) dx2[i] += dy[i];
                t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0); t0=t1;

                // dWo async (overlaps with SDPA backward)
                memcpy(do_out_buf, dx2, SEQ*DIM*4);
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, do_out_buf, SEQ, attn_out, SEQ, 1.0f, gWo, DIM);
                });

                // SDPA backward (ANE) — includes Wo^T conv
                io_copy(kSdpaBwd1->ioIn, 0, kFwdAttn->ioOut, DIM, 3*DIM, SEQ);
                io_write_fp16_at(kSdpaBwd1->ioIn, 3*DIM, dx2, DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(kSdpaBwd1);
                t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                io_copy(kSdpaBwd2->ioIn, 0, kSdpaBwd1->ioOut, DIM, 2*SCORE_CH, SEQ);
                io_copy(kSdpaBwd2->ioIn, 2*SCORE_CH, kFwdAttn->ioOut, DIM, 2*DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(kSdpaBwd2);
                t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;

                // Read dq,dk,dv — dW FFN+dWo still running async on serial queue
                io_read_fp16(kSdpaBwd2->ioOut, dq, 0,   DIM, SEQ);
                io_read_fp16(kSdpaBwd2->ioOut, dk, DIM,  DIM, SEQ);
                io_read_fp16(kSdpaBwd1->ioOut, dv, 0,    DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                // dWq/dWk/dWv queues after dWo on serial queue — no wait needed
                dispatch_group_async(dw_grp, dw_q, ^{
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, dq, SEQ, xnorm, SEQ, 1.0f, gWq, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, dk, SEQ, xnorm, SEQ, 1.0f, gWk, DIM);
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, DIM, DIM, SEQ,
                                1.0f, dv, SEQ, xnorm, SEQ, 1.0f, gWv, DIM);
                });

                // QKV backward (ANE) — dWq/dWk/dWv runs async
                io_copy(kQKVb->ioIn, 0, kSdpaBwd2->ioOut, 0, 2*DIM, SEQ);
                io_copy(kQKVb->ioIn, 2*DIM, kSdpaBwd1->ioOut, 0, DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;
                ane_eval(kQKVb);
                t1=mach_absolute_time(); t_ane+=tb_ms(t1-t0); t0=t1;
                io_read_fp16(kQKVb->ioOut, dx_attn, 0, DIM, SEQ);
                t1=mach_absolute_time(); t_io+=tb_ms(t1-t0); t0=t1;

                // RMSNorm1 backward (CPU) — doesn't touch cblas buffers
                float *dx_rms = calloc(SEQ*DIM, 4);
                rmsnorm_bwd(dx_rms, grms1, dx_attn, x_in, rms1_w, DIM, SEQ);
                free(dx_rms);
                t1=mach_absolute_time(); t_rms+=tb_ms(t1-t0);

                steps_batch++;
                if (step % 10 == 0 || step == start_step)
                    printf("step %-4d loss=%.6f\n", step, loss);
            }
            double tms = tb_ms(mach_absolute_time() - tt);
            total_train_ms += tms;
            total_steps_done += steps_batch;
            total_batches++;

            // Ensure all async dW finished before Adam
            dispatch_group_wait(dw_grp, DISPATCH_TIME_FOREVER);
            // Adam update (scale gradients by 1/steps_batch for averaging)
            float gsc = 1.0f / steps_batch;
            for(size_t i=0;i<wq_sz;i++){gWq[i]*=gsc;gWk[i]*=gsc;gWv[i]*=gsc;gWo[i]*=gsc;}
            for(size_t i=0;i<w1_sz;i++) gW1[i]*=gsc;
            for(size_t i=0;i<w2_sz;i++) gW2[i]*=gsc;
            for(size_t i=0;i<w3_sz;i++) gW3[i]*=gsc;
            for(int i=0;i<DIM;i++){grms1[i]*=gsc; grms2[i]*=gsc;}
            adam_t++;
            adam_update(Wq, gWq, &aWq, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(Wk, gWk, &aWk, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(Wv, gWv, &aWv, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(Wo, gWo, &aWo, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(W1, gW1, &aW1, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(W2, gW2, &aW2, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(W3, gW3, &aW3, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(rms1_w, grms1, &arms1, adam_t, lr, adam_b1, adam_b2, adam_eps);
            adam_update(rms2_w, grms2, &arms2, adam_t, lr, adam_b1, adam_b2, adam_eps);

            printf("  [batch %d: compile=%.0fms train=%.1fms (%.1fms/step) compiles=%d]\n",
                   steps_batch, cms, tms, tms/steps_batch, g_compile_count);
            printf("    ane=%.1f io=%.1f elem=%.1f rms=%.1f cblas_wait=%.1f ms/step\n",
                   t_ane/steps_batch, t_io/steps_batch, t_elem/steps_batch,
                   t_rms/steps_batch, t_cblas_wait/steps_batch);
        }

        // === Efficiency Report ===
        double wall = tb_ms(mach_absolute_time() - t_wall_start);
        total_compile_ms += cum_compile; total_train_ms += cum_train;
        wall += cum_wall; total_steps_done += cum_steps; total_batches += cum_batches;
        double fwd_flops = 4.0*2*DIM*DIM*SEQ + 2.0*2*DIM*HIDDEN*SEQ + 2.0*HIDDEN*DIM*SEQ;
        double bwd_dx_flops = fwd_flops;
        double sdpa_flops = 2.0*HEADS*5*SEQ*SEQ*HD;
        double ane_flops = (fwd_flops + bwd_dx_flops + sdpa_flops) * total_steps_done;
        double total_flops = (fwd_flops*3 + sdpa_flops) * total_steps_done; // fwd+bwd_dx+bwd_dw+sdpa

        printf("\n=== Efficiency Report ===\n");
        printf("Total steps:     %d\n", total_steps_done);
        printf("Wall time:       %.0f ms (%.1f s)\n", wall, wall/1000);
        printf("Compile time:    %.0f ms (%.1f%%)\n", total_compile_ms, 100*total_compile_ms/wall);
        printf("Train time:      %.0f ms (%.1f%%)\n", total_train_ms, 100*total_train_ms/wall);
        printf("Avg compile:     %.0f ms per batch (5 kernels)\n", total_compile_ms/total_batches);
        printf("Avg train:       %.1f ms/step\n", total_train_ms/total_steps_done);
        printf("ANE TFLOPS:      %.2f sustained\n", ane_flops / (total_train_ms * 1e9));
        printf("Total TFLOPS:    %.2f (ANE+CPU)\n", total_flops / (total_train_ms * 1e9));
        printf("ANE utilization: %.1f%% of 15.8 TFLOPS\n", 100*ane_flops/(total_train_ms*1e9)/15.8);
        printf("Params:          %.2fM  Weights: %.1fMB FP16\n", total_params/1e6, total_params*2.0/1e6);

        // Cleanup
        free_kern(kFwdAttn);free_kern(kFwdFFN);free_kern(kFFNBwd);
        free_kern(kSdpaBwd1);free_kern(kSdpaBwd2);free_kern(kQKVb);
        free(Wq);free(Wk);free(Wv);free(Wo);free(W1);free(W2);free(W3);
        free(rms1_w);free(rms2_w);free(x_in);free(y_tgt);
        free(xnorm);free(Q);free(K);free(V);free(attn_out);free(o_out);
        free(x2);free(x2norm);free(h1);free(h3);free(silu_out);free(ffn_out);free(y_out);
        free(dy);free(dffn);free(dh1);free(dh3);free(dx_ffn);free(dx2);
        free(do_out_buf);free(dattn);free(dq);free(dk);free(dv);free(dx_attn);
        free(probs_flat);free(dp_flat);
        free(gWq);free(gWk);free(gWv);free(gWo);free(gW1);free(gW2);free(gW3);
        free(grms1);free(grms2);
        adam_free(&aWq);adam_free(&aWk);adam_free(&aWv);adam_free(&aWo);
        adam_free(&aW1);adam_free(&aW2);adam_free(&aW3);adam_free(&arms1);adam_free(&arms2);
        unlink(CKPT_PATH);
    }
    return 0;
}
