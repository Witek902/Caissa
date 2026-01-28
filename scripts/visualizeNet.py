from PIL import Image
import struct
import math 
from math import ceil

filePath = "../build_bmi2/src/utils/eval.pnn"
marginSize = 1
headerSize = 64
numKingBuckets = 11

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b


def weightToColor(w):
    if w > 0:
        t = math.sqrt(1.0 - math.exp(-w / 256.0))
        return (int(lerp(0,255,t)), int(lerp(0,0,t)), int(lerp(0,0,t)))
    else:
        t = math.sqrt(1.0 - math.exp(w / 256.0))
        return (int(lerp(0,0,t)), int(lerp(0,80,t)), int(lerp(0,255,t)))


def inputIndexToCoords(index):

    offset = 8 + marginSize

    # white pawns
    if index < 1 * 64:
        return (0 * offset + index % 8, index / 8)
    # white knights
    if index < 2 * 64:
        index = index - (1 * 64)
        return (1 * offset + index % 8, index / 8)
    # white bishops
    if index < 3 * 64:
        index = index - (2 * 64)
        return (2 * offset + index % 8, index / 8)
    # white rooks
    if index < 4 * 64:
        index = index - (3 * 64)
        return (3 * offset + index % 8, index / 8)
    # white queens
    if index < 5 * 64:
        index = index - (4 * 64)
        return (4 * offset + index % 8, index / 8)
    # white king
    if index < 6 * 64:
        index = index - (5 * 64)
        return (5 * offset + index % 8, index / 8)

    # black pawns
    if index < 7 * 64:
        index = index - (6 * 64)
        return (6 * offset + index % 8, index / 8)
    # black knights
    if index < 8 * 64:
        index = index - (7 * 64)
        return (7 * offset + index % 8, index / 8)
    # black bishops
    if index < 9 * 64:
        index = index - (8 * 64)
        return (8 * offset + index % 8, index / 8)
    # black rooks
    if index < 10 * 64:
        index = index - (9 * 64)
        return (9 * offset + index % 8, index / 8)
    # black queens
    if index < 11 * 64:
        index = index - (10 * 64)
        return (10 * offset + index % 8, index / 8)
    # black king
    if index < 12 * 64:
        index = index - (11 * 64)
        return (11 * offset + index % 8, index / 8)
        
    return (0,0)


def roundUpToMultiple(number, multiple):
    return multiple * ceil(number / multiple)


def main():
    data = open(filePath, "rb").read()

    (magic, version,
     layerSize0, layerSize1, layerSize2, layerSize3,
     layerVariants0, layerVariants1, layerVariants2, layerVariants3) = struct.unpack("IIIIIIIIII", data[0:40])

    print("File version: " + str(version))
    print("Layer 0 size: " + str(layerSize0))
    print("Layer 1 size: " + str(layerSize1))
    print("Layer 1 variants: " + str(layerVariants1))

    accumulatorSize = int(layerSize1 / 2)

    rawViewImg = Image.new('RGB', (accumulatorSize,layerSize0), color='black')
    rawViewPixels = rawViewImg.load()

    imgWidth = (12 + layerVariants1) * (8 + marginSize) + marginSize
    imgHeight = accumulatorSize * (8 + marginSize) + marginSize
    boardViewImg = Image.new('RGB', (imgWidth, imgHeight), color='black')
    boardViewPixels = boardViewImg.load()

    kingBucket = 0

    for i in range(768):
        for j in range(accumulatorSize):
            dataOffset = headerSize + 2 * (accumulatorSize * i + j) + 2 * 12 * 64 * accumulatorSize * kingBucket
            (weight,) = struct.unpack("h", data[dataOffset:(dataOffset+2)])
            color = weightToColor(weight)
            (x,y) = inputIndexToCoords(i)
            rawViewPixels[j, i] = color
            boardViewPixels[marginSize + x, marginSize + j * (8 + marginSize) + y] = color

    lastLayerWeightsDataOffset = headerSize + 2 * accumulatorSize * (12 * 64 * numKingBuckets + 1)
    for variant in range(layerVariants1):
        for j in range(accumulatorSize):
            dataOffset = lastLayerWeightsDataOffset + variant * roundUpToMultiple(2 * (layerSize1 + 1), 64) + 2 * j
            (weight,) = struct.unpack("h", data[dataOffset:(dataOffset+2)])
            color = weightToColor(weight)
    
            xOffset = marginSize + (12 + variant) * (8 + marginSize)
            yOffset = marginSize + j * (8 + marginSize)
            for x in range(8):
                for y in range(8):
                    boardViewPixels[xOffset + x, yOffset + y] = color

    rawViewImg.save('rawView.png')
    boardViewImg.save('boardView.png')

if __name__ == "__main__":
    main()