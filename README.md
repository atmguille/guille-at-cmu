# My first day at CMU, according to Stable Diffusion

This repository contains the code I used to generate images of myself on my hypothetical first day at CMU, using Stable Diffusion.

For privacy reasons, I am not uploading the fine-tuned models nor the pictures of myself that I used to train them, as anyone could generate images of me using them. However, I am uploading the code I used to generate the images, as well as the images of CMU's School of Computer Science that I used to let the model know how it looks like (see [`cmu_cs_images`](cmu_cs_images/)).

## Environment setup

To start with, follow [Hugging Face's tutorial](https://huggingface.co/docs/diffusers/training/dreambooth) on fine-tuning DreamBooth to set up your local environment. Then, to avoid overfitting, you should train the model with a prior-preserving loss. For this purpose, I used the *person_ddim* dataset, which you can download as follows:

```bash
git clone https://github.com/djbielejeski/Stable-Diffusion-Regularization-Images-person_ddim.git

mkdir -p regularization_images/person_ddim
mv -v Stable-Diffusion-Regularization-Images-person_ddim/person_ddim/*.* regularization_images/person_ddim
```

## Fine-tuning

I fine-tuned the model in two stages. First, I trained it with ~30 pictures of myself and with prior-preserving loss, using `train_person.py`. Once the model was able to generate good-quality images of me, I fine-tuned this model with the 5 pictures of CMU's School of Computer Science that I uploaded to this repository, using `train_building.py`.

## Generating images

Once the model is fine-tuned, `inference.ipynb` can be used to generate images of me on my first day at CMU. The images generated by the model are saved in the `potential_images` directory.

## Fun fact

A variety of prompts were tried until reaching the desired results. At the beginning of this process, my generated face was very noisy for photo-realistic images. However, the quality of my generated face was very good when generating fictional images or in a cartoon-like style. Surprisingly, the quality of the face radically improved when I asked the model to generate images of a person looking similar to me. I guess there are some internal checks inside the model to avoid generating realistic images of people, in a effort to avoid DeepFakes. But it seems that one can get over these checks quite easily.
