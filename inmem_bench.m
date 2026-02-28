#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <mach/mach_time.h>
#import <IOSurface/IOSurface.h>

static mach_timebase_info_data_t g_tb;
static double ticksToMs(uint64_t t) { return (double)t * g_tb.numer / g_tb.denom / 1e6; }

double benchInMem(int ch, int sp) {
    @autoreleasepool {
        NSError *e = nil;
        NSString *path = [NSString stringWithFormat:@"/tmp/ane_sram_%dch_%dsp.mlpackage", ch, sp];
        NSURL *compiled = [MLModel compileModelAtURL:[NSURL fileURLWithPath:path] error:&e];
        if (e) return -1;

        NSData *milData = [[NSString stringWithContentsOfFile:
            [[compiled path] stringByAppendingPathComponent:@"model.mil"]
            encoding:NSUTF8StringEncoding error:nil] dataUsingEncoding:NSUTF8StringEncoding];
        NSData *weightBlob = [NSData dataWithContentsOfFile:
            [[compiled path] stringByAppendingPathComponent:@"weights/weight.bin"]];

        Class Desc = NSClassFromString(@"_ANEInMemoryModelDescriptor");
        Class IMM = NSClassFromString(@"_ANEInMemoryModel");
        Class AR = NSClassFromString(@"_ANERequest");
        Class AIO = NSClassFromString(@"_ANEIOSurfaceObject");

        NSDictionary *wdict = @{
            @"@model_path/weights/weight.bin": @{@"offset": @64, @"data": weightBlob}
        };
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            Desc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) return -2;
        id model = ((id(*)(Class,SEL,id))objc_msgSend)(IMM, @selector(inMemoryModelWithDescriptor:), desc);
        if (!model) return -3;

        id hexId = ((id(*)(id,SEL))objc_msgSend)(model, @selector(hexStringIdentifier));
        NSString *tmpDir = [NSTemporaryDirectory() stringByAppendingPathComponent:hexId];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[tmpDir stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[tmpDir stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        [weightBlob writeToFile:[tmpDir stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];

        BOOL ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(compileWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return -4; }

        ok = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
            model, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!ok) { [fm removeItemAtPath:tmpDir error:nil]; return -5; }

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
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);

        int iters = 50;
        uint64_t t0 = mach_absolute_time();
        for (int i = 0; i < iters; i++)
            ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
                model, @selector(evaluateWithQoS:options:request:error:), 21, @{}, req, &e);
        double ms = ticksToMs(mach_absolute_time() - t0) / iters;

        ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(model, @selector(unloadWithQoS:error:), 21, &e);
        CFRelease(ioIn); CFRelease(ioOut);
        [fm removeItemAtPath:tmpDir error:nil];
        return ms;
    }
}

int main() {
    mach_timebase_info(&g_tb);
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);

    printf("=== In-Memory ANE Benchmark ===\n\n");
    printf("%-12s %8s %10s %8s\n", "Config", "W (MB)", "ms/eval", "TFLOPS");
    printf("---------------------------------------------\n");

    int chs[] = {256, 512, 1024, 2048, 3072, 4096};
    int sps[] = {64,  64,  64,   64,   64,   64};
    for (int i = 0; i < 6; i++) {
        int ch = chs[i], sp = sps[i];
        double w_mb = (double)ch*ch*2/1024/1024;
        double gf = 2.0*ch*ch*sp/1e9;
        double ms = benchInMem(ch, sp);
        double tflops = (ms > 0) ? gf/ms : 0;
        if (ms > 0)
            printf("%4dch x%2dsp %7.1f  %8.3f ms %7.2f\n", ch, sp, w_mb, ms, tflops);
        else
            printf("%4dch x%2dsp %7.1f  FAIL(%.0f)\n", ch, sp, w_mb, ms);
    }
    return 0;
}
