from PIL import Image, ImageChops


def diff(org_path, prot_path):
    img1 = Image.open(org_path).convert('RGB')
    img2 = Image.open(prot_path).convert('RGB')

    diff = ImageChops.difference(img1, img2)

    extrema = diff.getextrema()
    for i in range(3):
        if extrema[i][1] > 0:
            multiplier = 255 / extrema[i][1]
            diff = ImageChops.multiply(diff, Image.new('RGB', diff.size, (int(multiplier),)*3))
            break

    diff.save("visual_evidence.png")

    print("done")




# diff("original.jpg", "1.png")


# diff("original.jpg", "2.png")


diff("original.jpg", "3.png")


# diff("original.jpg", "4.png")

