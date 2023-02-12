from PIL import Image
import matplotlib.pyplot as plt

def frame_with_annotation(frame_path):
    """
    Display of the selected frame and its annotations.

    Parameters
    ----------
    img_path : str
        The file location of the frame.
    ann_img : str
        The location of the corresponding annotations file to the frame.
    Returns
    -------
    -
    """
    try:
        img = Image.open(frame_path)
    except:
        print("Wrong frame name")
        return

    fig, ax = plt.subplots(1, figsize=(15, 15))

    # Display the image
    ax.imshow(img, cmap = 'gray')
    ax.grid(False)

    plt.show()
