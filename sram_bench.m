#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }
static id g_client;
static Class AM, AR, AIO;

double bench(const char *path, int ch, int sp) {
    @autoreleasepool {
        NSError *e = nil;
        NSURL *compiled = [MLModel compileModelAtURL:
            [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&e];
        if (e) return -1;
        id model = ((id(*)(Class,SEL,id,id))objc_msgSend)(AM, @selector(modelAtURL:key:), compiled, @"s");
        BOOL ok = ((BOOL(*)(id,SEL,id,id,NSUInteger,NSError**))objc_msgSend)(
            g_client, @selector(compileModel:options:qos:error:), model,
            @{@"kANEFModelType":@"kANEFModelMIL",@"kANEFNetPlistFilenameKey":@"model.mil"}, 21, &e);
        if (!ok) return -2;
        ok = ((BOOL(*)(id,SEL,id,id,NSUInteger,NSError**))objc_msgSend)(
            g_client, @selector(loadModel:options:qos:error:), model, @{}, 21, &e);
        if (!ok) return -3;

        NSUInteger bytes = ch * sp * 4; // FP32 input
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

        int iters = 30;
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
    AM = NSClassFromString(@"_ANEModel");
    AR = NSClassFromString(@"_ANERequest");
    AIO = NSClassFromString(@"_ANEIOSurfaceObject");

    printf("=== ANE SRAM Probe: 1x1 Conv with Increasing Weight Size ===\n\n");
    printf("%-25s %8s %8s %8s %10s %8s\n", "Config", "W (MB)", "Act(MB)", "Tot(MB)", "ms/eval", "TFLOPS");
    printf("--------------------------------------------------------------------------\n");

    typedef struct { int ch; int sp; } S;
    S sizes[] = {{256,64},{512,64},{1024,64},{2048,64},{3072,64},{4096,64},{5120,64},{6144,64},{8192,32}};

    for (int i = 0; i < 9; i++) {
        int ch = sizes[i].ch, sp = sizes[i].sp;
        double w_mb = (double)ch * ch * 2 / 1024 / 1024;  // FP16 weights
        double a_mb = (double)ch * sp * 2 / 1024 / 1024;   // FP16 activations
        double tot = w_mb + 2 * a_mb;
        double gflop = 2.0 * ch * ch * sp / 1e9;

        char path[256];
        snprintf(path, sizeof(path), "/tmp/ane_sram_%dch_%dsp.mlpackage", ch, sp);
        double ms = bench(path, ch, sp);

        double tflops = (ms > 0) ? gflop / ms : -1;
        char label[64];
        snprintf(label, sizeof(label), "%dch x %dsp", ch, sp);

        if (ms > 0)
            printf("%-25s %7.1f  %7.2f  %7.1f  %8.3f ms %7.2f\n", label, w_mb, a_mb, tot, ms, tflops);
        else
            printf("%-25s %7.1f  %7.2f  %7.1f  FAIL(%.0f)\n", label, w_mb, a_mb, tot, ms);
    }

    printf("\nLook for the performance cliff to estimate SRAM size.\n");
    return 0;
}
