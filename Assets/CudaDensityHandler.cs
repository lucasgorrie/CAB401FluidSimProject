using System;
using System.Runtime.InteropServices;
using UnityEngine;

public static class CudaDensityHandler
{

    const string PLUGIN = "density_plugin";

    [DllImport(PLUGIN)]
    public static extern int cuda_plugin_init();

    [DllImport(PLUGIN, CallingConvention = CallingConvention.Cdecl)]
    public static extern int calculate_density_cuda(
        IntPtr predicted_positions,  
        IntPtr keys,  
        IntPtr Offsets,      
        IntPtr CellCounts,
        IntPtr rhos,     
        IntPtr rhos_near, 
        int N,
        int grid_size_x, int grid_size_y, int grid_size_z,
        float R
    );

    public static int CalculateDensityUsingGPU(
        Vector3[] predictedPositions,
        uint[] keys,
        uint[] offsets,
        uint[] cellCounts,
        float[] rhos,
        float[] rhos_near,
        int gridX, int gridY, int gridZ,
        float R)
    {
        int N = predictedPositions.Length;
        GCHandle hp = GCHandle.Alloc(predictedPositions, GCHandleType.Pinned);
        GCHandle hk = GCHandle.Alloc(keys, GCHandleType.Pinned);
        GCHandle ho = GCHandle.Alloc(offsets, GCHandleType.Pinned);
        GCHandle hc = GCHandle.Alloc(cellCounts, GCHandleType.Pinned);
        GCHandle hr = GCHandle.Alloc(rhos, GCHandleType.Pinned);
        GCHandle hrn = GCHandle.Alloc(rhos_near, GCHandleType.Pinned);

        IntPtr pPred = hp.AddrOfPinnedObject();
        IntPtr pKeys = hk.AddrOfPinnedObject();
        IntPtr pOffsets = ho.AddrOfPinnedObject();
        IntPtr pCounts = hc.AddrOfPinnedObject();
        IntPtr pRhos = hr.AddrOfPinnedObject();
        IntPtr pRhosNear = hrn.AddrOfPinnedObject();

        int result = calculate_density_cuda(pPred, pKeys, pOffsets, pCounts, pRhos, pRhosNear, N, gridX, gridY, gridZ, R);

        // avoid memory leak //
        hrn.Free();
        hr.Free();
        hc.Free();
        ho.Free();
        hk.Free();
        hp.Free();

        return result;
    }
}
