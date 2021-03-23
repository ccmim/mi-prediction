'''
Developed by Andres Diaz-Pinto
This class does the following:
 - reads the image using sitk (output: sitk image)
 - set to identity the directions, zero the origin and 1 the spacing (output: sitk image)
 - apply gaussian filter to the sitk (output: sitk image)
 - apply the transformations. i.e. spatial_transformation_augmented (output: sitk image)
 - apply normalization and intensity transformations to resulting numpy array (output: numpy array)

'''
import numpy as np
import SimpleITK as sitk
import utils.io.image
from utils.sitk_np import np_to_sitk
from utils.sitk_image import resample
from transformations.spatial import translation, scale, rotation, deformation, flip, composite
from transformations.intensity.sitk.smooth import gaussian as gaussian_sitk
from transformations.intensity.np.normalize import normalize_robust
from transformations.intensity.np.shift_scale_clamp import ShiftScaleClamp
from transformations.intensity.np.gamma import change_gamma_unnormalized
from utils.random_class import float_uniform
from skimage.transform import resize


class TransformationGenerator(object):

    def __init__(self,
                 sitk_pixel_type=sitk.sitkInt16,
                 dim=3,
                 input_gaussian_sigma=1.0,
                 set_identity_spacing=False,
                 set_zero_origin=False,
                 set_identity_direction=False,
                 round_spacing_precision=None,
                 preprocessing=True,
                 output_size=None,
                 output_spacing=None,
                 interpolator='cubic',
                 resample_sitk_pixel_type=None,
                 resample_default_pixel_value=None,
                 np_pixel_type=np.float32,
                 img_type='label',
                 pixel_margin_ratio = 0.3,
                 training=True,
                 post_processing_np=True,
                 normalize = -1):
        """
        Initializer.
        :param sitk_pixel_type: sitk pixel type to which the loaded image will be converted to.
        :param dim: The dimension.
        :param input_gaussian_sigma: Sigma value for input smoothing.
        :param set_identity_spacing: If true, the spacing of the sitk image will be set to 1 for every dimension.
        :param set_zero_origin: If true, the origin of the sitk image will be set to 0 for every dimension.
        :param set_identity_direction: If true, the direction of the sitk image will be set to 1 for every dimension.
        :param round_spacing_precision: If > 0, spacing will be rounded to this precision (as in round(x, round_spacing_origin_direction))
        :param preprocessing: Function that will be called for preprocessing a loaded sitk image, i.e., sitk_image = preprocessing(sitk_image)
        :param output_size: The resampled output image size in sitk format ([x, y] or [x, y, z]).
        :param output_spacing: The resampled output spacing.
        :param interpolator: The sitk interpolator string that will be used. See utils.sitk_image.get_sitk_interpolator
                             for possible values.
        :param resample_sitk_pixel_type: The sitk output pixel type of the resampling operation.
        :param resample_default_pixel_value: The default pixel value of pixel values that are outside the image region.
        :param np_pixel_type: The output np pixel type.
        :param training: True for training sample.
        :param post_processing_np: If True, it post processes the numpy of the resampled sitk image. This function takes a np array as input and return a np array.
        :normalize: If -1, it normalizes the image between -1 and. If 0, it normalizes the image between 0 and 1
        """
        self.sitk_pixel_type = sitk_pixel_type
        self.input_gaussian_sigma = input_gaussian_sigma
        self.dim=dim
        self.set_identity_spacing = set_identity_spacing
        self.set_zero_origin = set_zero_origin
        self.set_identity_direction = set_identity_direction
        self.round_spacing_precision = round_spacing_precision
        self.preprocessing = False # self.intensity_preprocessing_mr
        self.output_size = output_size
        self.output_spacing = output_spacing
        self.interpolator = interpolator
        self.resample_sitk_pixel_type = resample_sitk_pixel_type
        self.resample_default_pixel_value = resample_default_pixel_value
        self.np_pixel_type = np_pixel_type
        self.img_type = img_type
        self.training = training
        self.post_processing_np = post_processing_np
        self.pixel_margin_ratio = pixel_margin_ratio
        self.normalize = normalize


    # The input is a frame
    def load_input(self, path_img, crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length):

        # reading the sitk image
        self.path_img = path_img
        # Loading the image
        image = self.load_and_preprocess(crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length)
        # Applying the transformations
        transformation = self.spatial_transformation()
        # Resampling the image
        output_image_sitk = self.get_resampled_images(image, transformation)
        # convert to np array
        output_image_np = utils.sitk_np.sitk_to_np(output_image_sitk, self.np_pixel_type)

        # output_image_np = self.get_np_image(output_image_sitk)

        # Postprocessing the numpy array
        if self.post_processing_np:
            output_image_np = self.intensity_postprocessing_mr(output_image_np)

        if self.normalize == -1:
            output_image_np = 2.0*(output_image_np - np.min(output_image_np))/(np.max(output_image_np) - np.min(output_image_np))-1.0  # Normalize between -1 and 1
        elif self.normalize == 0:
            output_image_np = (output_image_np - np.min(output_image_np))/(np.max(output_image_np) - np.min(output_image_np))  # Normalize between 0 and 1


        return output_image_np


    def load_image(self):
        """
        Loads an image from a given path. Throws an exception, if the image could not be loaded.
        :return: The loaded sitk image.
        """
        try:
            return utils.io.image.read(self.path_img, self.sitk_pixel_type)
        except:
            raise


    def cropper(self, image_load, crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length):

        # The size of margin, determined by the ratio we defined above
        pixel_margin = int(round(self.pixel_margin_ratio * roi_length + 0.001))

        # Image load has this format slices x width x height
        image_load = np.rollaxis(image_load, 2, 0)
        image_data = np.rollaxis(image_load, 2, 1)
        # Image data has this format height x slices x slices

        original_r_min = max(0, crop_r_min)
        original_r_max = min(image_data.shape[0] - 1, crop_r_max)
        original_c_min = max(0, crop_c_min)
        original_c_max = min(image_data.shape[1] - 1, crop_c_max)

        crop_image_data = image_data[original_r_min:(original_r_max + 1), original_c_min:(original_c_max + 1), :]
        # Resizing the image
        crop_image_data = resize(crop_image_data, (self.output_size[0], self.output_size[1]))

        crop_image_data = np.rollaxis(crop_image_data, 2, 0)
        crop_image_data = np.rollaxis(crop_image_data, 2, 1)

        # It is very important to keep this data format slices x width x height

        return crop_image_data


    def intensity_preprocessing_mr(self, image):
        """
        Intensity preprocessing function, working on the loaded sitk image, before resampling.
        :param image: The sitk image.
        :return: The preprocessed sitk image.
        """
        return gaussian_sitk(image, self.input_gaussian_sigma)

    def preprocess(self, image):
        """
        Processes the loaded image based on the given parameters of __init__(), i.e.,
        set_identity_spacing, set_zero_origin, set_identity_direction, preprocessing
        :param image: The loaded sitk image.
        :return: The processed sitk image.
        """
        if image is None:
            return image
        if self.set_identity_spacing:
            image.SetSpacing([1] * image.GetDimension())
        if self.set_zero_origin:
            image.SetOrigin([0] * image.GetDimension())
        if self.set_identity_direction:
            image.SetDirection(np.eye(image.GetDimension()).flatten())
        if self.round_spacing_precision is not None:
            image.SetSpacing([round(x, self.round_spacing_precision) for x in image.GetSpacing()])
        if self.preprocessing:
            image = self.preprocessing(image)
        return image

    def load_and_preprocess(self, crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length):
        """
        Loads an image for a given path_img and performs additional processing.
        :return: The loaded and processed sitk image.
        """
        # Loading original image
        original_image = self.load_image()
        # Getting numpy array of the image
        np_array = sitk.GetArrayFromImage(original_image)
        # Cropping original image
        cropped_image_np = self.cropper(np_array, crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length)
        # transforming corpped image into sitk class
        self.sitk_image = sitk.GetImageFromArray(cropped_image_np)
        self.image_size = self.sitk_image.GetSize()
        self.image_spacing = self.sitk_image.GetSpacing()
        # Set identity direction, zero origin, identity spacing
        image = self.preprocess(self.sitk_image)

        return image

    # Apply transformations depending of the usage. Localization, segmentation or regression
    def intensity_postprocessing_mr_random(self, image):
        """
        Intensity postprocessing for MR input. Random augmentation version.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(random_shift=0.2,
                               random_scale=0.4,
                               clamp_min=-1.0)(image)

    def intensity_postprocessing_mr(self, image):
        """
        Intensity postprocessing for MR input.
        :param image: The np input image.
        :return: The processed image.
        """
        image = normalize_robust(image)
        return ShiftScaleClamp(clamp_min=-1.0)(image)

    # For these functions: They also depend on whether we want to do localization or segmentation
    # These transformation are done so the image is always inside the output space. i.e. translation.InputCenterToOrigin
    # and translation.OriginToOutputCenter keep the image inside the output space
    def spatial_transformation_augmented(self):
        """
        The spatial image transformation with random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        transformation_list = []
        transformation_list.extend([  # Translation transformation which transforms the input image center to the origin
                                    translation.InputCenterToOrigin(self.dim),
                                    # A translation transformation with a random offset.
                                    translation.Random(self.dim, [1, 1, 1]),
                                    # A rotation transformation with random angles (in radian)
                                    rotation.Random(self.dim, [0.10, 0.10, 0.10]),
                                    # A flip transformation with a random probability
                                    flip.Random(self.dim, [0.1, 0.1, 0.1]),
                                    # A scale transformation with a random scaling factor, equal for each dimension.
                                    scale.RandomUniform(self.dim, 0.1),
                                    # A scale transformation with random scaling factors
                                    scale.Random(self.dim, [0.05, 0.05, 0.05]),
                                    # Translation transformation which transforms origin to the the output image center
                                    translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing),
                                    # A deformation transformation in the output image physical domain
                                    deformation.Output(self.dim, [4, 4, 4], 2, self.image_size, self.image_spacing)
                                    ])
        return composite.Composite(self.dim, transformation_list, name='image').get(image=self.sitk_image)

    def spatial_transformation(self):
        """
        The spatial image transformation without random augmentation.
        :param datasources: datasources dict.
        :return: The transformation.
        """
        # print('This is image_size: ' + str(self.image_size))
        # print('This is image_spacing: ' + str(self.image_spacing))
        # print('This is sitk_image size: ' + str(sitk.GetArrayFromImage(self.sitk_image).shape))

        transformation_list = []
        transformation_list.append(translation.InputCenterToOrigin(self.dim))
        transformation_list.append(translation.OriginToOutputCenter(self.dim, self.image_size, self.image_spacing))
        return composite.Composite(self.dim, transformation_list, name='image').get(image=self.sitk_image)


    # These functions are defined in generators/image_generator

    def get_resampled_image(self, image, transformation):
        """
        Transforms the given sitk image with the given transformation.
        :param images: The sitk image.
        :param transformation: The sitk transformation.
        :return: The resampled sitk image.
        """
        output_image = resample(image,
                                transformation,
                                self.output_size,
                                self.output_spacing,
                                interpolator=self.interpolator,
                                output_pixel_type=self.resample_sitk_pixel_type,
                                default_pixel_value=self.resample_default_pixel_value)
        return output_image

    def get_resampled_images(self, images, transformation):
        """
        Transforms the given sitk image (or list of sitk images) with the given transformation.
        :param images: The sitk image (or list of sitk images).
        :param transformation: The sitk transformation.
        :return: The resampled sitk image (or list of sitk images).
        """
        if isinstance(images, list) or isinstance(images, tuple):
            return [self.get_resampled_image(image, transformation) for image in images]
        else:
            return self.get_resampled_image(images, transformation)

    def get(self, path_img, crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length):
        """
        :param path_img: path where the image is located.
        :return: The resampled np array.
        """

        # utils.io.image.write(image, 'image_loaded_preprocessed.nii.gz')

        if self.training:

            print('\n NOT IMPLEMENTED')

            # # This is not debugged!
            # if self.img_type == 'input':
            #
            #     # Applying the transformations
            #     transformation = self.spatial_transformation_augmented()
            #     # Resampling the image
            #     output_image_sitk = self.get_resampled_images(image, transformation)
            #     # convert to np array
            #     output_image_np = utils.sitk_np.sitk_to_np(output_image_sitk, self.np_pixel_type)
            #     # Postprocessing the numpy array
            #     if self.post_processing_np:
            #         output_image_np = self.intensity_postprocessing_mr_random(output_image_np)
            #
            # elif self.img_type == 'label':
            #
            #     # Applying the transformations
            #     transformation = self.spatial_transformation_augmented()
            #     # Resampling the image
            #     output_image_sitk = self.get_resampled_images(image, transformation)
            #     # convert to np array
            #     output_image_np = utils.sitk_np.sitk_to_np(output_image_sitk, self.np_pixel_type)
            #     # Postprocessing the numpy array
            #     if self.post_processing_np:
            #         output_image_np = self.intensity_postprocessing_mr_random(output_image_np)

        else:

            output_image_np = self.load_input(path_img, crop_c_min, crop_c_max, crop_r_min, crop_r_max, roi_length)

        return output_image_np
