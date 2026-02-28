#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static id g_client; static Class AM, AR, AIO;

double bench(const char *path, int ch, int sp) {
    @autoreleasepool {
        NSError *e = nil;
        NSURL *compiled = [MLModel compileModelAtURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&e];
        if (e) return -1;
        id model = ((id(*)(Class,SEL,id,id))objc_msgSend)(AM, @selector(modelAtURL:key:), compiled, @"s");
        ((BOOL(*)(id,SEL,id,id,NSUInteger,NSError**))objc_msgSend)(
            g_client, @selector(compileModel:options:qos:error:), model,
            @{@"kANEFModelType":@"kANEFModelMIL",@"kANEFNetPlistFilenameKey":@"model.mil"}, 21, &e);
        ((BOOL(*)(id,SEL,id,id,NSUInteger,NSError**))objc_msgSend)(
            g_client, @selector(loadModel:options:qos:error:), model, @{}, 21, &e);
        NSUInteger bytes = ch * sp * 4;
        IOSurfaceRef ioIn = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        IOSurfaceRef ioOut = IOSurfaceCreate((__bridge CFDictionaryRef)@{
            (id)kIOSurfaceWidth:@(bytes),(id)kIOSurfaceHeight:@1,
            (id)kIOSurfaceBytesPerElement:@1,(id)kIOSurfaceBytesPerRow:@(bytes),
            (id)kIOSurfaceAllocSize:@(bytes),(id)kIOSurfacePixelFormat:@0});
        id wIn = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioIn);
        id wOut = ((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(AIO, @selector(objectWithIOSurface:), ioOut);
        id req = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(AR,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            @[wIn], @[@0], @[wOut], @[@0], nil, nil, @0);
        for (int i = 0; i < 5; i++)
            ((BOOL(*)(id,SEL,id,id,id,NSUInteger,NSError**))objc_msgSend)(
                g_client, @selector(evaluateWithModel:options:request:qos:error:), model, @{}, req, 21, &e);
        int iters = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,id,id,id,NSUInteger,NSError**))objc_msgSend)(
                g_client, @selector(evaluateWithModel:options:request:qos:error:), model, @{}, req, 21, &e);
        double ms = ticksToMs(mach_absolute_time() - t0) / iters;
        ((void(*)(id,SEL,id,id,NSUInteger,NSError**))objc_msgSend)(
            g_client, @selector(unloadModel:options:qos:error:), model, @{}, 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        return ms;
    }
}

int main() {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    g_client = [NSClassFromString(@"_ANEClient") performSelector:@selector(sharedConnection)];
    AM = NSClassFromString(@"_ANEModel"); AR = NSClassFromString(@"_ANERequest");
    AIO = NSClassFromString(@"_ANEIOSurfaceObject");

    printf("=== ANE SRAM Fine Probe (weights only vary, spatial=64) ===\n\n");
    printf("%-12s %8s %10s %8s %12s\n", "Channels", "W (MB)", "ms/eval", "TFLOPS", "GFLOPS/MB");
    printf("--------------------------------------------------------------\n");

    int chs[] = {256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5120, 6144, 8192};
    int sps[] = {64,  64,  64,   64,   64,   64,   64,   64,   64,   64,   64,   64,   32};

    for (int i = 0; i < 13; i++) {
        int ch = chs[i], sp = sps[i];
        double w_mb = (double)ch * ch * 2 / 1024 / 1024;
        double gf = 2.0 * ch * ch * sp / 1e9;
        char path[256];
        snprintf(path, sizeof(path), "/tmp/ane_sram_%dch_%dsp.mlpackage", ch, sp);
        double ms = bench(path, ch, sp);
        double tf = (ms > 0) ? gf / ms : 0;
        double eff = (ms > 0) ? tf * 1000 / w_mb : 0;
        printf("%6d ch   %7.1f  %8.3f ms %7.2f  %10.1f %s\n",
               ch, w_mb, ms, tf, eff,
               (i > 0 && eff < 100) ? " <-- spilling?" : "");
    }
    return 0;
}
