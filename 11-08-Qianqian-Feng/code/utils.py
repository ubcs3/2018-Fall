import numpy as np
from scipy.sparse import csr_matrix as sparse_matrix
from scipy import sparse
import csv





def create_recipe_ingredients_matrix(data,recipe_id="id",ingredients_key="ingredients"):

    
    recipe_ids = list(recipe[recipe_id] for recipe in data)
    ingredients = list(set([i for recipe in data for i in recipe[ingredients_key]]))
    cuisines = list(set([recipe["cuisine"] for recipe in data]))

    n = len(data)
    d = len(ingredients)
    k = len(cuisines)
    print("Number of recipes:",n)
    print("Number of ingredients:",d)
    print("Different kinds of cuisines:",k)

    # {'recipe-name': index}
    # {'ingredient-name': index}
    recipe_mapper = dict(zip(recipe_ids, list(range(n))))
    ingredient_mapper = dict(zip(ingredients, list(range(d))))
    cuisine_mapper = dict(zip(cuisines, list(range(k))))

    # {index: 'recipe-name'}
    # {index: 'ingredient-name'}
    recipe_inverse_mapper = dict(zip(list(range(n)), recipe_ids))
    ingredient_inverse_mapper = dict(zip(list(range(d)), ingredients))
    cuisines_inverse_mapper = dict(zip(range(k), cuisines))

    X = sparse_matrix([[int(ingredient_inverse_mapper[i] in recipe[ingredients_key]) for i in range(d)] for recipe in data])
    y = np.array([cuisine_mapper[recipe["cuisine"]] for recipe in data])

    sparse.save_npz('../data/train.npz', X)
    with open('../data/ingredients.csv','w') as outfile:
        writer = csv.DictWriter(outfile, ingredient_mapper.keys())
        writer.writeheader()
        writer.writerow(ingredient_mapper)

    
def load(data,recipe_id="id",ingredients_key="ingredients"):

    recipe_ids = list(recipe[recipe_id] for recipe in data)
    cuisines = list(set([recipe["cuisine"] for recipe in data]))

    with open('../data/ingredients.csv','rt') as file:
        reader = csv.DictReader(file)
        for row in reader:
            ingredients = row.keys()
            ingredient_mapper = row

    file.close()

    n = len(data)
    d = len(ingredients)
    k = len(cuisines)
    print("Number of recipes:",n)
    print("Number of ingredients:",d)
    print("Different kinds of cuisines:",k)

    # {'recipe-name': index}
    # {'ingredient-name': index}
    recipe_mapper = dict(zip(recipe_ids, list(range(n))))
    
    cuisine_mapper = dict(zip(cuisines, list(range(k))))

    # {index: 'recipe-name'}
    # {index: 'ingredient-name'}
    recipe_inverse_mapper = dict(zip(list(range(n)), recipe_ids))
    ingredient_inverse_mapper = dict(zip(list(range(d)), ingredients))
    cuisines_inverse_mapper = dict(zip(range(k), cuisines))

    X = sparse.load_npz('../data/train.npz')
    y = np.array([cuisine_mapper[recipe["cuisine"]] for recipe in data])

    return X, y, recipe_mapper, ingredient_mapper, recipe_inverse_mapper, ingredient_inverse_mapper, cuisine_mapper, cuisines_inverse_mapper        













