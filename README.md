# get-self-attention

Do you understand Transformers? No? Yes? This might help either way.

The `try-it.ipynb` notebook lets you

- generate some simple synthetic data,
- create and train very tweakable, minimal models,
- visualise some of the model parameters as it learns.

## Running this code

Two options:

- Run `try-it.ipynb` if you have an adequate environment.
- Run `docker compose up` if you have Docker.

## Recommended exercise

Generate a simple data set,
e.g.: `text = generate_deterministic_sequence(length=3, n=100, whitespace=True)`. 

Then create a
minimal model, e.g.: `n_embd = 1, n_head = 1, head_size = 1, block_size = 1` and visualise it. 

Try to answer the following questions:
- Will the model learn the given sequence?
- What is the "smallest" model that can learn it?
