float3 readPoint(TensorView<float> points, int pointIndex)
{
    return float3(
        points[pointIndex, 0],
        points[pointIndex, 1],
        points[pointIndex, 2]
    );
}

float3 readTriangleVertex(TensorView<float> triangles, int triangleIndex, int vertexIndex)
{
    return float3(
        triangles[triangleIndex, vertexIndex, 0],
        triangles[triangleIndex, vertexIndex, 1],
        triangles[triangleIndex, vertexIndex, 2]
    );
}

[Differentiable]
float safeSignedDenom(float value, float eps)
{
    return value >= 0.0 ? max(value, eps) : min(value, -eps);
}

[Differentiable]
float pointTriangleDistanceSquared(float3 point, float3 a, float3 b, float3 c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = point - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        return dot(ap, ap);
    }

    float3 bp = point - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        return dot(bp, bp);
    }

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        float v = d1 / (d1 - d3);
        float3 closest = a + v * ab;
        float3 delta = point - closest;
        return dot(delta, delta);
    }

    float3 cp = point - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        return dot(cp, cp);
    }

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        float w = d2 / safeSignedDenom(d2 - d6, 1.0e-12);
        float3 closest = a + w * ac;
        float3 delta = point - closest;
        return dot(delta, delta);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        float3 bc = c - b;
        float w = (d4 - d3) / safeSignedDenom((d4 - d3) + (d5 - d6), 1.0e-12);
        float3 closest = b + w * bc;
        float3 delta = point - closest;
        return dot(delta, delta);
    }

    float denom = safeSignedDenom(va + vb + vc, 1.0e-12);
    float vFace = vb / denom;
    float wFace = vc / denom;
    float3 closest = a + ab * vFace + ac * wFace;
    float3 delta = point - closest;
    return dot(delta, delta);
}

[Differentiable]
float pointTriangleUnsignedDistance(float3 point, float3 a, float3 b, float3 c)
{
    const float smoothEps = 1.0e-12;
    float distanceSquared = pointTriangleDistanceSquared(point, a, b, c);
    return sqrt(max(distanceSquared, 0.0) + smoothEps) - sqrt(smoothEps);
}

[Differentiable]
float triangleSolidAngle(float3 point, float3 a0, float3 b0, float3 c0)
{
    const float eps = 1.0e-12;

    float3 a = a0 - point;
    float3 b = b0 - point;
    float3 c = c0 - point;

    float la = max(length(a), eps);
    float lb = max(length(b), eps);
    float lc = max(length(c), eps);
    float numerator = dot(a, cross(b, c));
    float denominator =
        la * lb * lc
        + dot(a, b) * lc
        + dot(b, c) * la
        + dot(c, a) * lb;
    return 2.0 * atan2(numerator, safeSignedDenom(denominator, eps));
}

void addTriangleGradient(
    TensorView<float> gradTriangles,
    int triangleIndex,
    float3 gradA,
    float3 gradB,
    float3 gradC
)
{
    float oldValue = 0.0;

    gradTriangles.InterlockedAdd(uint3(triangleIndex, 0, 0), gradA.x, oldValue);
    gradTriangles.InterlockedAdd(uint3(triangleIndex, 0, 1), gradA.y, oldValue);
    gradTriangles.InterlockedAdd(uint3(triangleIndex, 0, 2), gradA.z, oldValue);

    gradTriangles.InterlockedAdd(uint3(triangleIndex, 1, 0), gradB.x, oldValue);
    gradTriangles.InterlockedAdd(uint3(triangleIndex, 1, 1), gradB.y, oldValue);
    gradTriangles.InterlockedAdd(uint3(triangleIndex, 1, 2), gradB.z, oldValue);

    gradTriangles.InterlockedAdd(uint3(triangleIndex, 2, 0), gradC.x, oldValue);
    gradTriangles.InterlockedAdd(uint3(triangleIndex, 2, 1), gradC.y, oldValue);
    gradTriangles.InterlockedAdd(uint3(triangleIndex, 2, 2), gradC.z, oldValue);
}

float3 readNodeBBoxMin(TensorView<float> bboxMin, int nodeIndex)
{
    return float3(
        bboxMin[nodeIndex, 0],
        bboxMin[nodeIndex, 1],
        bboxMin[nodeIndex, 2]
    );
}

float3 readNodeBBoxMax(TensorView<float> bboxMax, int nodeIndex)
{
    return float3(
        bboxMax[nodeIndex, 0],
        bboxMax[nodeIndex, 1],
        bboxMax[nodeIndex, 2]
    );
}

float pointAabbDistanceSquared(float3 point, float3 bboxMin, float3 bboxMax)
{
    float dx = point.x < bboxMin.x ? (bboxMin.x - point.x) : (point.x > bboxMax.x ? point.x - bboxMax.x : 0.0);
    float dy = point.y < bboxMin.y ? (bboxMin.y - point.y) : (point.y > bboxMax.y ? point.y - bboxMax.y : 0.0);
    float dz = point.z < bboxMin.z ? (bboxMin.z - point.z) : (point.z > bboxMax.z ? point.z - bboxMax.z : 0.0);
    return dx * dx + dy * dy + dz * dz;
}

bool rayAabbHit(float3 origin, float3 direction, float3 bboxMin, float3 bboxMax)
{
    const float eps = 1.0e-12;
    float3 invDirection = float3(
        abs(direction.x) > eps ? 1.0 / direction.x : 1.0e30,
        abs(direction.y) > eps ? 1.0 / direction.y : 1.0e30,
        abs(direction.z) > eps ? 1.0 / direction.z : 1.0e30
    );

    float tx1 = (bboxMin.x - origin.x) * invDirection.x;
    float tx2 = (bboxMax.x - origin.x) * invDirection.x;
    float tmin = min(tx1, tx2);
    float tmax = max(tx1, tx2);

    float ty1 = (bboxMin.y - origin.y) * invDirection.y;
    float ty2 = (bboxMax.y - origin.y) * invDirection.y;
    tmin = max(tmin, min(ty1, ty2));
    tmax = min(tmax, max(ty1, ty2));

    float tz1 = (bboxMin.z - origin.z) * invDirection.z;
    float tz2 = (bboxMax.z - origin.z) * invDirection.z;
    tmin = max(tmin, min(tz1, tz2));
    tmax = min(tmax, max(tz1, tz2));

    return tmax >= max(tmin, 0.0);
}

bool rayTriangleHit(float3 origin, float3 direction, float3 a, float3 b, float3 c, out float t)
{
    const float epsilon = 1.0e-6;

    float3 edge1 = b - a;
    float3 edge2 = c - a;
    float3 h = cross(direction, edge2);
    float determinant = dot(edge1, h);

    if (abs(determinant) < epsilon)
    {
        t = 0.0;
        return false;
    }

    float invDeterminant = 1.0 / determinant;
    float3 s = origin - a;
    float u = invDeterminant * dot(s, h);
    if (u < 0.0 || u > 1.0)
    {
        t = 0.0;
        return false;
    }

    float3 q = cross(s, edge1);
    float v = invDeterminant * dot(direction, q);
    if (v < 0.0 || (u + v) > 1.0)
    {
        t = 0.0;
        return false;
    }

    t = invDeterminant * dot(edge2, q);
    return t > epsilon;
}

[AutoPyBindCUDA]
[CUDAKernel]
void queryMeshUnsignedDistance(
    TensorView<float> triangles,
    TensorView<float> points,
    TensorView<float> unsignedDistance,
    TensorView<int> closestTriangleIndex
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int pointIndex = int(dispatchIdx.x);

    if (pointIndex >= unsignedDistance.size(0))
    {
        return;
    }

    const float smoothEps = 1.0e-12;
    float3 point = readPoint(points, pointIndex);
    float minDistanceSquared = 1.0e30;
    int closestTriangle = -1;

    for (int triangleIndex = 0; triangleIndex < triangles.size(0); ++triangleIndex)
    {
        float3 a = readTriangleVertex(triangles, triangleIndex, 0);
        float3 b = readTriangleVertex(triangles, triangleIndex, 1);
        float3 c = readTriangleVertex(triangles, triangleIndex, 2);
        float distanceSquared = pointTriangleDistanceSquared(point, a, b, c);
        if (distanceSquared < minDistanceSquared)
        {
            minDistanceSquared = distanceSquared;
            closestTriangle = triangleIndex;
        }
    }

    unsignedDistance[pointIndex] = sqrt(max(minDistanceSquared, 0.0) + smoothEps) - sqrt(smoothEps);
    closestTriangleIndex[pointIndex] = closestTriangle;
}

[AutoPyBindCUDA]
[CUDAKernel]
void queryMeshDistanceAndWinding(
    TensorView<float> triangles,
    TensorView<float> points,
    TensorView<float> unsignedDistance,
    TensorView<float> windingAngle,
    TensorView<int> closestTriangleIndex
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int pointIndex = int(dispatchIdx.x);

    if (pointIndex >= unsignedDistance.size(0))
    {
        return;
    }

    const float smoothEps = 1.0e-12;
    float3 point = readPoint(points, pointIndex);
    float minDistanceSquared = 1.0e30;
    float totalAngle = 0.0;
    int closestTriangle = -1;

    for (int triangleIndex = 0; triangleIndex < triangles.size(0); ++triangleIndex)
    {
        float3 a = readTriangleVertex(triangles, triangleIndex, 0);
        float3 b = readTriangleVertex(triangles, triangleIndex, 1);
        float3 c = readTriangleVertex(triangles, triangleIndex, 2);
        float distanceSquared = pointTriangleDistanceSquared(point, a, b, c);
        if (distanceSquared < minDistanceSquared)
        {
            minDistanceSquared = distanceSquared;
            closestTriangle = triangleIndex;
        }
        totalAngle += triangleSolidAngle(point, a, b, c);
    }

    unsignedDistance[pointIndex] = sqrt(max(minDistanceSquared, 0.0) + smoothEps) - sqrt(smoothEps);
    windingAngle[pointIndex] = totalAngle;
    closestTriangleIndex[pointIndex] = closestTriangle;
}

[AutoPyBindCUDA]
[CUDAKernel]
void queryMeshUnsignedDistanceBVH(
    TensorView<float> triangles,
    TensorView<float> points,
    TensorView<float> nodeBBoxMin,
    TensorView<float> nodeBBoxMax,
    TensorView<int> nodeLeft,
    TensorView<int> nodeRight,
    TensorView<int> nodeStart,
    TensorView<int> nodeCount,
    TensorView<int> triangleIndices,
    TensorView<float> unsignedDistance,
    TensorView<int> closestTriangleIndex
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int pointIndex = int(dispatchIdx.x);

    if (pointIndex >= unsignedDistance.size(0))
    {
        return;
    }

    const float smoothEps = 1.0e-12;
    const int maxStackDepth = 64;
    int stack[maxStackDepth];
    int stackSize = 0;
    stack[stackSize++] = 0;

    float3 point = readPoint(points, pointIndex);
    float minDistanceSquared = 1.0e30;
    int closestTriangle = -1;

    while (stackSize > 0)
    {
        int nodeIndex = stack[--stackSize];
        float3 bboxMin = readNodeBBoxMin(nodeBBoxMin, nodeIndex);
        float3 bboxMax = readNodeBBoxMax(nodeBBoxMax, nodeIndex);
        float nodeDistanceSquared = pointAabbDistanceSquared(point, bboxMin, bboxMax);
        if (nodeDistanceSquared > minDistanceSquared)
        {
            continue;
        }

        int count = nodeCount[nodeIndex];
        if (count > 0)
        {
            int start = nodeStart[nodeIndex];
            for (int offset = 0; offset < count; ++offset)
            {
                int triangleIndex = triangleIndices[start + offset];
                float3 a = readTriangleVertex(triangles, triangleIndex, 0);
                float3 b = readTriangleVertex(triangles, triangleIndex, 1);
                float3 c = readTriangleVertex(triangles, triangleIndex, 2);
                float distanceSquared = pointTriangleDistanceSquared(point, a, b, c);
                if (distanceSquared < minDistanceSquared)
                {
                    minDistanceSquared = distanceSquared;
                    closestTriangle = triangleIndex;
                }
            }
            continue;
        }

        int leftChild = nodeLeft[nodeIndex];
        int rightChild = nodeRight[nodeIndex];
        if (leftChild < 0 || rightChild < 0)
        {
            continue;
        }

        float leftDistanceSquared = pointAabbDistanceSquared(point, readNodeBBoxMin(nodeBBoxMin, leftChild), readNodeBBoxMax(nodeBBoxMax, leftChild));
        float rightDistanceSquared = pointAabbDistanceSquared(point, readNodeBBoxMin(nodeBBoxMin, rightChild), readNodeBBoxMax(nodeBBoxMax, rightChild));

        if (leftDistanceSquared < rightDistanceSquared)
        {
            if (stackSize + 2 <= maxStackDepth)
            {
                stack[stackSize++] = rightChild;
                stack[stackSize++] = leftChild;
            }
        }
        else
        {
            if (stackSize + 2 <= maxStackDepth)
            {
                stack[stackSize++] = leftChild;
                stack[stackSize++] = rightChild;
            }
        }
    }

    unsignedDistance[pointIndex] = sqrt(max(minDistanceSquared, 0.0) + smoothEps) - sqrt(smoothEps);
    closestTriangleIndex[pointIndex] = closestTriangle;
}

[AutoPyBindCUDA]
[CUDAKernel]
void queryMeshParitySignBVH(
    TensorView<float> triangles,
    TensorView<float> points,
    TensorView<float> nodeBBoxMin,
    TensorView<float> nodeBBoxMax,
    TensorView<int> nodeLeft,
    TensorView<int> nodeRight,
    TensorView<int> nodeStart,
    TensorView<int> nodeCount,
    TensorView<int> triangleIndices,
    float jitterScale,
    TensorView<int> inside
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int pointIndex = int(dispatchIdx.x);

    if (pointIndex >= inside.size(0))
    {
        return;
    }

    const int maxStackDepth = 64;
    int stack[maxStackDepth];
    int stackSize = 0;
    stack[stackSize++] = 0;

    float3 point = readPoint(points, pointIndex);
    float3 direction = float3(1.0, 0.0, 0.0);
    float3 origin = point + float3(0.0, jitterScale, jitterScale * 1.618034);
    int intersections = 0;

    while (stackSize > 0)
    {
        int nodeIndex = stack[--stackSize];
        float3 bboxMin = readNodeBBoxMin(nodeBBoxMin, nodeIndex);
        float3 bboxMax = readNodeBBoxMax(nodeBBoxMax, nodeIndex);
        if (!rayAabbHit(origin, direction, bboxMin, bboxMax))
        {
            continue;
        }

        int count = nodeCount[nodeIndex];
        if (count > 0)
        {
            int start = nodeStart[nodeIndex];
            for (int offset = 0; offset < count; ++offset)
            {
                int triangleIndex = triangleIndices[start + offset];
                float3 a = readTriangleVertex(triangles, triangleIndex, 0);
                float3 b = readTriangleVertex(triangles, triangleIndex, 1);
                float3 c = readTriangleVertex(triangles, triangleIndex, 2);
                float t;
                if (rayTriangleHit(origin, direction, a, b, c, t))
                {
                    intersections += 1;
                }
            }
            continue;
        }

        int leftChild = nodeLeft[nodeIndex];
        int rightChild = nodeRight[nodeIndex];
        if (leftChild >= 0 && stackSize < maxStackDepth)
        {
            stack[stackSize++] = leftChild;
        }
        if (rightChild >= 0 && stackSize < maxStackDepth)
        {
            stack[stackSize++] = rightChild;
        }
    }

    inside[pointIndex] = (intersections & 1) != 0 ? 1 : 0;
}

[AutoPyBindCUDA]
[CUDAKernel]
void backwardMeshUnsignedDistance(
    TensorView<float> triangles,
    TensorView<float> points,
    TensorView<int> closestTriangleIndex,
    TensorView<float> gradUnsignedDistance,
    TensorView<float> gradTriangles,
    TensorView<float> gradPoints
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int pointIndex = int(dispatchIdx.x);

    if (pointIndex >= gradUnsignedDistance.size(0))
    {
        return;
    }

    float gradOutput = gradUnsignedDistance[pointIndex];
    if (gradOutput == 0.0)
    {
        return;
    }

    int triangleIndex = closestTriangleIndex[pointIndex];
    if (triangleIndex < 0)
    {
        return;
    }

    float3 point = readPoint(points, pointIndex);
    float3 a = readTriangleVertex(triangles, triangleIndex, 0);
    float3 b = readTriangleVertex(triangles, triangleIndex, 1);
    float3 c = readTriangleVertex(triangles, triangleIndex, 2);

    DifferentialPair<float3> dpPoint = DifferentialPair<float3>(point, float3(0.0));
    DifferentialPair<float3> dpA = DifferentialPair<float3>(a, float3(0.0));
    DifferentialPair<float3> dpB = DifferentialPair<float3>(b, float3(0.0));
    DifferentialPair<float3> dpC = DifferentialPair<float3>(c, float3(0.0));
    bwd_diff(pointTriangleUnsignedDistance)(dpPoint, dpA, dpB, dpC, gradOutput);

    addTriangleGradient(gradTriangles, triangleIndex, dpA.d, dpB.d, dpC.d);
    gradPoints[pointIndex, 0] = dpPoint.d.x;
    gradPoints[pointIndex, 1] = dpPoint.d.y;
    gradPoints[pointIndex, 2] = dpPoint.d.z;
}

[AutoPyBindCUDA]
[CUDAKernel]
void backwardMeshDistanceAndWinding(
    TensorView<float> triangles,
    TensorView<float> points,
    TensorView<int> closestTriangleIndex,
    TensorView<float> gradUnsignedDistance,
    TensorView<float> gradWindingAngle,
    TensorView<float> gradTriangles,
    TensorView<float> gradPoints
)
{
    uint3 dispatchIdx = cudaThreadIdx() + cudaBlockIdx() * cudaBlockDim();
    int pointIndex = int(dispatchIdx.x);

    if (pointIndex >= gradUnsignedDistance.size(0))
    {
        return;
    }

    float gradUnsigned = gradUnsignedDistance[pointIndex];
    float gradAngle = gradWindingAngle[pointIndex];
    if (gradUnsigned == 0.0 && gradAngle == 0.0)
    {
        return;
    }

    float3 point = readPoint(points, pointIndex);
    float3 pointGradient = float3(0.0);

    if (gradAngle != 0.0)
    {
        for (int triangleIndex = 0; triangleIndex < triangles.size(0); ++triangleIndex)
        {
            float3 a = readTriangleVertex(triangles, triangleIndex, 0);
            float3 b = readTriangleVertex(triangles, triangleIndex, 1);
            float3 c = readTriangleVertex(triangles, triangleIndex, 2);

            DifferentialPair<float3> dpPoint = DifferentialPair<float3>(point, float3(0.0));
            DifferentialPair<float3> dpA = DifferentialPair<float3>(a, float3(0.0));
            DifferentialPair<float3> dpB = DifferentialPair<float3>(b, float3(0.0));
            DifferentialPair<float3> dpC = DifferentialPair<float3>(c, float3(0.0));
            bwd_diff(triangleSolidAngle)(dpPoint, dpA, dpB, dpC, gradAngle);

            addTriangleGradient(gradTriangles, triangleIndex, dpA.d, dpB.d, dpC.d);
            pointGradient += dpPoint.d;
        }
    }

    if (gradUnsigned != 0.0)
    {
        int triangleIndex = closestTriangleIndex[pointIndex];
        if (triangleIndex >= 0)
        {
            float3 a = readTriangleVertex(triangles, triangleIndex, 0);
            float3 b = readTriangleVertex(triangles, triangleIndex, 1);
            float3 c = readTriangleVertex(triangles, triangleIndex, 2);

            DifferentialPair<float3> dpPoint = DifferentialPair<float3>(point, float3(0.0));
            DifferentialPair<float3> dpA = DifferentialPair<float3>(a, float3(0.0));
            DifferentialPair<float3> dpB = DifferentialPair<float3>(b, float3(0.0));
            DifferentialPair<float3> dpC = DifferentialPair<float3>(c, float3(0.0));
            bwd_diff(pointTriangleUnsignedDistance)(dpPoint, dpA, dpB, dpC, gradUnsigned);

            addTriangleGradient(gradTriangles, triangleIndex, dpA.d, dpB.d, dpC.d);
            pointGradient += dpPoint.d;
        }
    }

    gradPoints[pointIndex, 0] = pointGradient.x;
    gradPoints[pointIndex, 1] = pointGradient.y;
    gradPoints[pointIndex, 2] = pointGradient.z;
}
