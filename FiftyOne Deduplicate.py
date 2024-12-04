import pathlib
import sys
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz
from fiftyone import ViewField as F
from sklearn.metrics.pairwise import cosine_similarity


## CONFIGURE

# Note: holostrawberry recommends 0.97 to 0.99.
similarity_threshold = 0.96

##

images_folder = pathlib.Path(sys.argv[1]).resolve()
model_name = "clip-vit-base32-torch"
supported_types = (".png", ".jpg", ".jpeg")

img_count = len(list(images_folder.iterdir()))
batch_size = min(250, img_count)

non_images = [
    f for f in images_folder.iterdir() if not f.suffix.lower().endswith(supported_types)
]
if non_images:
    print(
        f"💥 Error: Found non-image file {non_images[0]} - This program doesn't allow it. Sorry! Use the Extras at the bottom to clean the folder."
    )
elif img_count == 0:
    print(f"💥 Error: No images found in {images_folder}")
else:
    print("\n💿 Analyzing dataset...\n")
    dataset = fo.Dataset.from_dir(images_folder, dataset_type=fo.types.ImageDirectory)
    model = foz.load_zoo_model(model_name)
    embeddings = dataset.compute_embeddings(model, batch_size=batch_size)

    batch_embeddings = np.array_split(embeddings, batch_size)
    similarity_matrices = []
    max_size_x = max(array.shape[0] for array in batch_embeddings)
    max_size_y = max(array.shape[1] for array in batch_embeddings)

    for i, batch_embedding in enumerate(batch_embeddings):
        similarity = cosine_similarity(batch_embedding)
        # Pad 0 for np.concatenate
        padded_array = np.zeros((max_size_x, max_size_y))
        padded_array[0 : similarity.shape[0], 0 : similarity.shape[1]] = similarity
        similarity_matrices.append(padded_array)

    similarity_matrix = np.concatenate(similarity_matrices, axis=0)
    similarity_matrix = similarity_matrix[
        0 : embeddings.shape[0], 0 : embeddings.shape[0]
    ]

    similarity_matrix = cosine_similarity(embeddings)
    similarity_matrix -= np.identity(len(similarity_matrix))

    dataset.match(F("max_similarity") > similarity_threshold)
    dataset.tags = ["delete", "has_duplicates"]

    id_map = [s.id for s in dataset.select_fields(["id"])]
    samples_to_remove = set()
    samples_to_keep = set()

    for idx, sample in enumerate(dataset):
        if sample.id not in samples_to_remove:
            # Keep the first instance of two duplicates
            samples_to_keep.add(sample.id)

            dup_idxs = np.where(similarity_matrix[idx] > similarity_threshold)[0]
            for dup in dup_idxs:
                # We kept the first instance so remove all other duplicates
                samples_to_remove.add(id_map[dup])

            if len(dup_idxs) > 0:
                sample.tags.append("has_duplicates")
                sample.save()
        else:
            sample.tags.append("delete")
            sample.save()

    # clear_output()

    sidebar_groups = fo.DatasetAppConfig.default_sidebar_groups(dataset)
    for group in sidebar_groups[1:]:
        group.expanded = False
    dataset.app_config.sidebar_groups = sidebar_groups
    dataset.save()
    session = fo.launch_app(dataset)

    print("❗ Wait a minute for the session to load. If it doesn't, read above.")
    print("❗ When it's ready, you'll see a grid of your images.")
    print(
        '❗ On the left side enable "sample tags" to visualize the images marked for deletion.'
    )
    print(
        '❗ You can mark your own images with the "delete" label by selecting them and pressing the tag icon at the top.'
    )
    input("⭕ When you're done, enter something here to save your changes: ")

    session.refresh()
    fo.close_app()
    # clear_output()
    print("💾 Saving...")
    kys = [s for s in dataset if "delete" in s.tags]
    dataset.delete_samples(kys)
    export = images_folder.parent / f"{images_folder.stem}_export"
    export = export.resolve()
    export.mkdir(exist_ok=True)
    dataset.export(export_dir=str(export), dataset_type=fo.types.ImageDirectory)
