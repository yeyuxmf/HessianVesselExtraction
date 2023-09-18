
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

def Hessian2D(I, sigma):
    # I = torch.tensor(I).view(1, 1, I.shape[0], I.shape[1]).float()

    H_arange = np.arange(-3*sigma, 3*sigma+1)
    W_arange = np.arange(-3*sigma, 3*sigma+1)
    Y, X = np.meshgrid(W_arange, H_arange, indexing='ij')
    X2 = np.power(X, 2)
    Y2 = np.power(Y, 2)
    DGaxx = 1/ (2 * np.pi*(sigma**4)) * (X2 / (sigma**2) -1) * np.exp(-(X2 + Y2) / (2 * (sigma**2)))
    DGaxy = 1/ (2 * np.pi*(sigma**6)) * (X * Y) * np.exp(-(X2 + Y2) / (2 * (sigma**2)))
    DGayy = DGaxx.T


    HessThin = hessian_matrix(I, sigma)
    I = cv2.GaussianBlur(I, (2 * sigma + 1, 2 * sigma + 1), sigma)
    Ixx = cv2.filter2D(I, -1, DGaxx, borderType=cv2.BORDER_REPLICATE)
    Ixy = cv2.filter2D(I, -1, DGaxy, borderType=cv2.BORDER_REPLICATE)
    Iyy = cv2.filter2D(I, -1, DGayy, borderType=cv2.BORDER_REPLICATE)

    return Ixx, Ixy, Iyy

def eig2image(Ixx, Ixy, Iyy):
    diff_xy = Ixx-Iyy
    tmp = np.sqrt(np.power(diff_xy, 2) + 4 * np.power(Ixy, 2))


    # Compute the eigenvalues
    mu1 = 0.5 * (Ixx + Iyy + tmp)
    mu2 = 0.5 * (Ixx + Iyy - tmp)
    mask = np.abs(mu1) < np.abs(mu2)

    Lambda1 = mu1.copy()
    Lambda1[mask] = mu2[mask]
    Lambda2 = mu2.copy()
    Lambda2[mask] = mu1[mask]
    mu1[mask] = mu2[mask]


    return Lambda2, Lambda1


def customFrangiFilter2D(img, options):
    eps = 0.0000001
    sigmas = [ i for i in range(options["FrangiScaleRange"][0], options["FrangiScaleRange"][1], options["FrangiScaleRatio"])]

    beta = 2 * options["FrangiBetaOne"] ** 2
    c = 2 * options["FrangiBetaTwo"] ** 2

    ALLfiltered = np.zeros((len(sigmas), img.shape[0], img.shape[1]))
    for si in range(len(sigmas)):
        if options["verbose"]:
            print("Current Frangi Filter Sigma: ", sigmas[si])

        Ixx, Ixy, Iyy = Hessian2D(img/np.max(img), sigmas[si])
        Ixx = (sigmas[si] ** 2) * Ixx
        Ixy = (sigmas[si] ** 2) * Ixy
        Iyy = (sigmas[si] ** 2) * Iyy

        Lambda2, Lambda1 = eig2image(Ixx, Ixy, Iyy)

        Lambda1[Lambda1 == 0] = eps
        Rb = np.power(Lambda2 / Lambda1, 2)
        S2 = np.power(Lambda1,2) + np.power(Lambda2, 2)

        Ifiltered = np.exp(-Rb / beta) * (np.ones_like(img) - np.exp(-S2 / c))

        if options["BlackWhite"]:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        ALLfiltered[si,:,:] = Ifiltered

    outIm = np.max(ALLfiltered, axis=0)

    return outIm



def skimageFrangiFilter2D(img, options):
    eps = 0.0000001
    sigmas = [ i for i in range(options["FrangiScaleRange"][0], options["FrangiScaleRange"][1], options["FrangiScaleRatio"])]

    beta = 2 * options["FrangiBetaOne"] ** 2
    c = 2 * options["FrangiBetaTwo"] ** 2

    ALLfiltered = np.zeros((len(sigmas), img.shape[0], img.shape[1]))
    for si in range(len(sigmas)):
        if options["verbose"]:
            print("Current Frangi Filter Sigma: ", sigmas[si])

        imgG = cv2.GaussianBlur(img, (2*sigmas[si]+1, 2*sigmas[si]+1), sigmas[si])
        HessThin = hessian_matrix(imgG, sigmas[si])
        EignThin = hessian_matrix_eigvals(HessThin)
        Lambda2, Lambda1 = EignThin[1], EignThin[0]


        Lambda1[Lambda1 == 0] = eps
        Rb = np.power(Lambda2 / Lambda1, 2)
        S2 = np.power(Lambda1,2) + np.power(Lambda2, 2)

        Ifiltered = np.exp(-Rb / beta) * (np.ones_like(img) - np.exp(-S2 / c))


        if options["BlackWhite"]:
            Ifiltered[Lambda1 < 0] = 0
        else:
            Ifiltered[Lambda1 > 0] = 0

        ALLfiltered[si,:,:] = Ifiltered

    outIm = np.max(ALLfiltered, axis=0)

    return outIm





import cv2
if __name__ == "__main__":

    file_path ="./data/04_test.tif"
    img = cv2.imread(file_path)[:, :, 1]



    options = {'FrangiScaleRange': [1, 10], 'FrangiScaleRatio': 2, 'FrangiBetaOne': 0.5, 'FrangiBetaTwo': 15,
                     'verbose': True, 'BlackWhite': True}

    #Customized Gaussian kernel for Hessian matrix and vessel extraction.
    c_outIm = customFrangiFilter2D(img, options)
    #Utilize the Hessian matrix provided by the skimage package to extract blood vessels.
    sk_outIm = skimageFrangiFilter2D(img, options)


    c_outIm = c_outIm.astype(np.float32) / np.max(c_outIm)
    sk_outIm = sk_outIm.astype(np.float32) / np.max(sk_outIm)


    cv2.namedWindow("c_outIm", cv2.WINDOW_NORMAL)
    cv2.imshow("c_outIm", c_outIm)
    cv2.namedWindow("sk_outIm", cv2.WINDOW_NORMAL)
    cv2.imshow("sk_outIm", sk_outIm)

    cv2.waitKey(0)




