import numpy
import cv2
from .transforms import rgb2ntsc,ntsc2rgb


def create_gaussian_image_pyramid(image, pyramid_levels):
    gauss_copy = numpy.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(1, pyramid_levels):
        gauss_copy = cv2.pyrDown(gauss_copy)
        img_pyramid.append(gauss_copy)

    return img_pyramid

def gaussianDwn(image, pyramid_levels):
    gauss_copy = numpy.ndarray(shape=image.shape, dtype="float")
    gauss_copy[:] = image
    img_pyramid = [gauss_copy]
    for pyramid_level in range(1, pyramid_levels+1):
        gauss_copy = cv2.pyrDown(gauss_copy)
        img_pyramid.append(gauss_copy)

    return img_pyramid

def create_laplacian_image_pyramid(image, pyramid_levels):
    gauss_pyramid = create_gaussian_image_pyramid(image, pyramid_levels)
    laplacian_pyramid = []
    for i in range(pyramid_levels - 1):
        laplacian_pyramid.append((gauss_pyramid[i] - cv2.pyrUp(gauss_pyramid[i + 1], dstsize=(gauss_pyramid[i].shape[1],gauss_pyramid[i].shape[0]))) + 0)

    laplacian_pyramid.append(gauss_pyramid[-1])
    return laplacian_pyramid


def create_gaussian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, gaussianDwn)


def create_laplacian_video_pyramid(video, pyramid_levels):
    return _create_pyramid(video, pyramid_levels, create_laplacian_image_pyramid)


def _create_pyramid(video, pyramid_levels, pyramid_fn):
    vid_pyramid = []
    # frame_count, height, width, colors = video.shape
    for frame_number, frame in enumerate(video):
        frame = rgb2ntsc(frame.astype(numpy.float32))
        frame_pyramid = pyramid_fn(frame, pyramid_levels)

        for pyramid_level, pyramid_sub_frame in enumerate(frame_pyramid):
            if frame_number == 0:
                vid_pyramid.append(
                    numpy.zeros((video.shape[0], pyramid_sub_frame.shape[0], pyramid_sub_frame.shape[1], 3),
                                dtype="float"))

            vid_pyramid[pyramid_level][frame_number] = pyramid_sub_frame

    return vid_pyramid


def collapse_laplacian_pyramid(image_pyramid):
    img = ntsc2rgb(image_pyramid.pop())
    while image_pyramid:
        x = (ntsc2rgb(image_pyramid.pop())- 0)
        img = cv2.pyrUp(img,dstsize = (x.shape[1],x.shape[0])) + x
    return img


def collapse_gaussian_pyramid(image_pyramid,level):
    image_pyramid.pop()
    img = image_pyramid.pop() - 0
    while image_pyramid:
        last = image_pyramid.pop() - 0
    final = ntsc2rgb(last + cv2.resize(img,(last.shape[1],last.shape[0])))/2
    final[final > 1] = 1
    final[final < 0] = 0
    return final

def collapse_gaussian_video_pyramid(pyramid,level):
    i = 0
    while True:
        try:
            img_pyramid = [vid[i] for vid in pyramid]
            pyramid[0][i] = collapse_gaussian_pyramid(img_pyramid,level)
            i += 1
        except IndexError:
            break
    return pyramid[0]

def collapse_laplacian_video_pyramid(pyramid):
    i = 0
    while True:
        try:
            img_pyramid = [vid[i] for vid in pyramid]
            pyramid[0][i] = collapse_laplacian_pyramid(img_pyramid)
            i += 1
        except IndexError:
            break
    return pyramid[0]
