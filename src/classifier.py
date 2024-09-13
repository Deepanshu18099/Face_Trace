"""An example of how to use your own dataset to train a classifier that recognizes people.
"""
# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections
import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Connect to Milvus
connections.connect()

from pymilvus import (
    FieldSchema, CollectionSchema, DataType, Collection
)

MATCHES_TO_SHOW = 10 # during classification, show the top 3 matches
MAX_CLUSTERS = 5 # maximum number of clusters to consider for each person
MIN_EXAMPLES_TO_CLUSTER = 4 # minimum number of examples to consider for clustering


def create_collection(collection_name, embedding_size):
    # Define fields for the schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Primary key
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=embedding_size),  # Embedding vector
        FieldSchema(name="label", dtype=DataType.INT64)  # Class label
    ]
    
    # Create the schema with the primary key and fields
    schema = CollectionSchema(fields, description="Face Embeddings Collection")
    
    # Create the collection
    collection = Collection(name=collection_name, schema=schema)
    return collection

def drop_collection(collectionName):
    # To drop the collection, mean to remove the collection and its data
    try:
        collection = Collection(collectionName)
        collection.drop()
        print("Collection dropped")
    except Exception as e:
        print("Error:", e)

def search_embeddings(collection, query_embedding, top_k=5):
    search_params = {"metric_type": "L2", "params": {"nprobe": 16}}
    results = collection.search([query_embedding], "embedding", limit=top_k, param=search_params, output_fields=["label"])
    return results

def create_index(collection):
    # Index parameters for creating the index
    index_params = {
        "metric_type": "L2",  # Similarity metric: L2 for Euclidean distance
        "index_type": "IVF_FLAT",  # Index type: IVF_FLAT is simple for testing
        "params": {"nlist": 128}  # Number of clusters
    }
    
    # Create the index on the 'embedding' field
    print("Creating index on 'embedding' field")
    collection.create_index(field_name="embedding", index_params=index_params)

def cluster_embeddings(embeddings, max_clusters=MAX_CLUSTERS, min_clusters=1):
    """
    Cluster embeddings for each person and return centroids.
    The number of clusters is dynamically chosen based on variability.
    
    Args:
    - embeddings: List or array of face embeddings for a person.
    - max_clusters: Maximum number of clusters to consider.
    - min_clusters: Minimum number of clusters to allow.
    
    Returns:
    - centroids: Centroid of the clusters representing distinct variations.
    """
    best_clusters = min_clusters
    best_score = -2  # Silhouette score ranges from -1 to 1
    centroids = None

    # if there are less than 4 embeddings, then take number of clusters as number of embeddings
    if len(embeddings) < MIN_EXAMPLES_TO_CLUSTER:
        best_clusters = len(embeddings)
        return embeddings
    else:
        # Iterate over a range of clusters to find the best fit
        for n_clusters in range(min_clusters, min(max_clusters+1, len(embeddings)+1)):
            kmeans = KMeans(n_clusters=n_clusters)
            cluster_labels = kmeans.fit_predict(embeddings)

            print(f"Cluster labels: {cluster_labels}, size of embeddings: {len(embeddings)}, size_clusters: {n_clusters}")
            # Calculate a silhouette score to evaluate how well the clusters represent the data
            if(n_clusters == 1 or n_clusters == len(embeddings)):
                score = 0
            else:
                score = silhouette_score(embeddings, cluster_labels)

            print(f"Clusters: {n_clusters}, Score: {score:.4f}")
            if score > best_score:
                best_score = score
                best_clusters = n_clusters
                centroids = kmeans.cluster_centers_
    
    print(f"Selected {best_clusters} clusters for {len(embeddings)} images based on silhouette score: {best_score:.4f}")
    return centroids

def insert_cluster_centroids(collection, embeddings, labels, max_clusters=MAX_CLUSTERS):
    """
    Insert only the cluster centroids into the database, reducing the number of embeddings.
    
    Args:
    - collection: Milvus collection.
    - embeddings: Dictionary of embeddings grouped by person (person_id -> embeddings).
    - labels: List of person IDs corresponding to the embeddings.
    - max_clusters: Maximum number of clusters to store for each person.
    """
    for person_id, person_embeddings in embeddings.items():
        # Cluster the embeddings and get the centroids
        centroids = cluster_embeddings(person_embeddings, max_clusters=max_clusters)

        # print(f"Centroids: {centroids}")
        
        # Insert the centroids into the collection
        data = [centroids, [person_id] * len(centroids)]  # Repeated labels for the centroids
        # label should be such that same person will have same person ids in the data
        # so we need to store these person Ids some how, these are basically from embeddings
        res = collection.insert(data)
        print(f"Inserted {len(centroids)} centroids for person ID {person_id}")
    
    create_index(collection)


# def insert_embeddings(collection, embeddings, labels):
#     data = [embeddings, labels]
#     res = collection.insert(data)
#     print("Inserted {} embeddings".format(len(embeddings)))
#     return res

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            np.random.seed(seed=args.seed)
            
            if args.use_split_dataset:
                dataset_tmp = facenet.get_dataset(args.data_dir)
                print('Number of classes: %d' % len(dataset_tmp))
                train_set, test_set = split_dataset(dataset_tmp, args.min_nrof_images_per_class, args.nrof_train_images_per_class)
                if (args.mode=='TRAIN'):
                    dataset = train_set
                elif (args.mode=='CLASSIFY'):
                    dataset = test_set
            else:
                dataset = facenet.get_dataset(args.data_dir)

            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')            

                 
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            # for all the images in the dataset, if same folder then same label
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(args.model)
            
            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Create a collection in Milvus
            collection_name = "face_embeddings"

            
            # Run forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            
            # classifier_filename_exp = os.path.expanduser(args.classifier_filename)
# Now as model is almost done, next thing will be create a new python model for recognisation In which we will give some image path, so fi So my model and architecture is ready for recognisation,
# Now I want to make a Portal for the model, In which
# Backend possible flask: 
# will be running tensorflow and docker of milvus
# - will recieve the image from frontend using some API and 
            # if (args.mode=='TRAIN'):
            #     # Train classifier
            #     print('Training classifier')
            #     model = SVC(kernel='linear', probability=True)
            #     model.fit(emb_array, labels)
            
            #     # Create a list of class names
            #     class_names = [ cls.name.replace('_', ' ') for cls in dataset]

            #     # Saving classifier model
            #     with open(classifier_filename_exp, 'wb') as outfile:
            #         pickle.dump((model, class_names), outfile)
            #     print('Saved classifier model to file "%s"' % classifier_filename_exp)
                
            # elif (args.mode=='CLASSIFY'):
            #     # Classify images
            #     print('Testing classifier')
            #     with open(classifier_filename_exp, 'rb') as infile:
            #         (model, class_names) = pickle.load(infile)
            #     print('Loaded classifier model from file "%s"' % classifier_filename_exp)
            #     predictions = model.predict_proba(emb_array)
            #     # best_class_indices = np.argmax(predictions, axis=1)
            #     # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
            #     # for i in range(len(best_class_indices)):
            #     #     print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
            #     # printing best 2 classes
            #     best_class_indices = np.argsort(predictions, axis=1)[:, -2:]
            #     best1_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices[:, -1]]
            #     best2_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices[:, -2]]
            #     for i in range(len(best_class_indices)):
            #         # also print the item name
            #         print('%4d %s %s: %.3f, %s: %.3f' % (i, paths[i].split('/')[-1], class_names[best_class_indices[i, -1]], best1_class_probabilities[i], class_names[best_class_indices[i, -2]], best2_class_probabilities[i]))
            #     # accuracy = np.mean(np.equal(best_class_indices, labels))
            #     # print('Accuracy: %.3f' % accuracy)

            # if args.mode == 'TRAIN':
            #     # dropprevious collection
            #     drop_collection(collection_name)
            #     # Create a collection in Milvus
            #     collection = create_collection(collection_name, embedding_size)
                
            #     # Insert embeddings into Milvus
            #     print('Inserting embeddings into Milvus')
            #     # print(labels[1], "--------------------")
            #     insert_embeddings(collection, emb_array, labels)

            if args.mode == 'TRAIN':
                drop_collection(collection_name)
                collection = create_collection(collection_name, embedding_size)
                # print("Collection: ", collection)
                
                # Organize embeddings by person label
                embeddings_by_person = {}

                # print size of labels and emb_array for verification
                print('Size of labels:', len(labels))
                print('Size of emb_array:', len(emb_array))
                for i, label in enumerate(labels):
                    if label not in embeddings_by_person:
                        embeddings_by_person[label] = []
                    embeddings_by_person[label].append(emb_array[i])

                print('Number of people:', len(embeddings_by_person))
                print('Inserting clustered centroids into Milvus')

                # Insert clustered centroids into Milvus
                insert_cluster_centroids(collection, embeddings_by_person, labels)

            elif args.mode == 'CLASSIFY':
                # Load collection
                collection = Collection(collection_name)
                collection.load()

                class_names = [ cls.name.replace('_', ' ') for cls in dataset]

                # Search for nearest neighbors in Milvus
                print('Classifying images')
                best_class_indices = []
                for i in range(nrof_images):
                    query_embedding = emb_array[i]
                    # load collection
                    results = search_embeddings(collection, query_embedding, top_k=MATCHES_TO_SHOW)
                    for result in results:
                        best_class_indices.append(result[0].entity.get("label"))
                        print(f"Image: {paths[i].split('/')[-1][:10]}")
                        for i in range(MATCHES_TO_SHOW):
                            print(f"--------{class_names[result[i].label]}-{result[i].distance}")
                        
                              # print(f"---------------------- also close to class {class_names[result[1].label]} with distance {result[1].distance}")
                # calculating accuracy of best 1 class
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
            
def split_dataset(dataset, min_nrof_images_per_class, nrof_train_images_per_class):
    train_set = []
    test_set = []
    # Image_name_test = []
    for cls in dataset:
        paths = cls.image_paths
        # print('Shape of paths:', len(paths))
        # Remove classes with less than min_nrof_images_per_class
        if len(paths)>=min_nrof_images_per_class:
            np.random.shuffle(paths)
            train_set.append(facenet.ImageClass(cls.name, paths[:nrof_train_images_per_class]))
            test_set.append(facenet.ImageClass(cls.name, paths[nrof_train_images_per_class:]))
            # # push all the elements in the list
            # Image_name_test.append(paths[nrof_train_images_per_class:])
            # print('Shape of Image_name_test:', len(test_set))
    return train_set, test_set

            
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
        help='Indicates if a new classifier should be trained or a classification ' + 
        'model should be used for classification', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('model', type=str, 
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename', 
        help='Classifier model file name as a pickle (.pkl) file. ' + 
        'For training this is the output and for classification this is an input.')
    parser.add_argument('--use_split_dataset', 
        help='Indicates that the dataset specified by data_dir should be split into a training and test set. ' +  
        'Otherwise a separate test set can be specified using the test_data_dir option.', action='store_true')
    parser.add_argument('--test_data_dir', type=str,
        help='Path to the test data directory containing aligned images used for testing.')
    parser.add_argument('--batch_size', type=int,
        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
        help='Random seed.', default=666)
    parser.add_argument('--min_nrof_images_per_class', type=int,
        help='Only include classes with at least this number of images in the dataset', default=20)
    parser.add_argument('--nrof_train_images_per_class', type=int,
        help='Use this number of images from each class for training and the rest for testing', default=10)
    
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
