from PIL import Image
import struct
import math 

filePath = "../data/neuralNets/eval-9.pnn"
marginSize = 1
headerSize = 64

def lerp(a: float, b: float, t: float) -> float:
    return (1 - t) * a + t * b


def weightToColor(w):
    black = 10
    if w > 0:
        t = 1.0 - math.exp(-w / 128.0)
        return (int(lerp(black,255,t)), int(lerp(black,10,t)), int(lerp(black,20,t)))
    else:
        t = 1.0 - math.exp(w / 128.0)
        return (int(lerp(black,10,t)), int(lerp(black,80,t)), int(lerp(black,255,t)))


def inputIndexToCoords(index):

    offset = 8 + marginSize

    # white pawns
    if index < 48:
        return (0 * offset + index % 8, 1 + index / 8)
    # white knights
    if index < 48 + 64:
        index = index - (48)
        return (1 * offset + index % 8, index / 8)
    # white bishops
    if index < 48 + 2 * 64:
        index = index - (48 + 1 * 64)
        return (2 * offset + index % 8, index / 8)
    # white rooks
    if index < 48 + 3 * 64:
        index = index - (48 + 2 * 64)
        return (3 * offset + index % 8, index / 8)
    # white queens
    if index < 48 + 4 * 64:
        index = index - (48 + 3 * 64)
        return (4 * offset + index % 8, index / 8)
    # white king
    if index < 48 + 4 * 64 + 32:
        index = index - (48 + 4 * 64)
        return (5 * offset + index % 4, index / 4)
    
    index = index - (48 + 4 * 64 + 32)

    # black pawns
    if index < 48:
        return (6 * offset + index % 8, 1 + index / 8)
    # black knights
    if index < 48 + 64:
        index = index - (48)
        return (7 * offset + index % 8, index / 8)
    # black bishops
    if index < 48 + 2 * 64:
        index = index - (48 + 1 * 64)
        return (8 * offset + index % 8, index / 8)
    # black rooks
    if index < 48 + 3 * 64:
        index = index - (48 + 2 * 64)
        return (9 * offset + index % 8, index / 8)
    # black queens
    if index < 48 + 4 * 64:
        index = index - (48 + 3 * 64)
        return (10 * offset + index % 8, index / 8)
    # black king
    if index < 48 + 5 * 64:
        index = index - (48 + 4 * 64)
        return (11 * offset + index % 8, index / 8)
    
    return (0,0)


def main():
    data = open(filePath, "rb").read()

    (magic, version, layerSize0, layerSize1) = struct.unpack("IIII", data[0:16])

    print("File version: " + str(version))
    print("Layer 0 size: " + str(layerSize0))
    print("Layer 1 size: " + str(layerSize1))

    rawViewImg = Image.new('RGB', (layerSize1,layerSize0), color='black')
    rawViewPixels = rawViewImg.load()

    imgWidth = 13 * (8 + marginSize) + marginSize
    imgHeight = layerSize1 * (8 + marginSize) + marginSize
    boardViewImg = Image.new('RGB', (imgWidth, imgHeight), color='black')
    boardViewPixels = boardViewImg.load()

    for i in range(layerSize0):
        for j in range(layerSize1):
            dataOffset = headerSize + 2 * (layerSize1 * i + j)
            (weight,) = struct.unpack("h", data[dataOffset:(dataOffset+2)])
            color = weightToColor(weight)
            (x,y) = inputIndexToCoords(i)
            rawViewPixels[j, i] = color
            boardViewPixels[marginSize + x, marginSize + j * (8 + marginSize) + y] = color

    # last layer weights
    for j in range(layerSize1):
        dataOffset = headerSize + 2 * (layerSize1 * (layerSize0 + 1)) + 2 * j
        (weight,) = struct.unpack("h", data[dataOffset:(dataOffset+2)])
        color = weightToColor(weight)
        for x in range(8):
            for y in range(8):
                boardViewPixels[marginSize + 12 * (8 + marginSize) + x, marginSize + j * (8 + marginSize) + y] = color

    rawViewImg.save('rawView.png')
    boardViewImg.save('boardView.png')

if __name__ == "__main__":
    main()