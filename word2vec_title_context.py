# coding=utf-8
# A modified version of Word2Vec TensorFlow implementation
# (github.com/tensorflow/tensorflow/tree/r0.11/tensorflow/examples/tutorials/word2vec)
#
# According to Stanford 224d Course
import collections
import math
import os
import random
import zipfile

import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

# Step 1: Download the data.
url = 'http://mattmahoney.net/dc/'

# def maybe_download(filename, expected_bytes):
#   """Download a file if not present, and make sure it's the right size."""
#   if not os.path.exists(filename):
#     filename, _ = urllib.request.urlretrieve(url + filename, filename)
#   statinfo = os.stat(filename)
#   if statinfo.st_size == expected_bytes:
#     print('Found and verified', filename)
#   else:
#     print(statinfo.st_size)
#     raise Exception(
#         'Failed to verify ' + filename + '. Can you get to it with a browser?')
#   return filename
#
# filename = maybe_download('text8.zip', 31344016)

filename="sample_title_doc.txt"
vecname="vector.bin"
dicname="dict"
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with open(filename,'r') as f:
    data = f.read().replace("###"," ").split()
  return data

words = read_data(filename)
print('Data size', len(words))
# Step 2: Build the dictionary and replace rare words with UNK token.
blockLine=100#shuffle frequency
batch_size = 1000
num_steps = 1600000 #
embedding_size = 64  # embedding length
vocabulary_size = 100000
dictionary = dict()
def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0

  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
fw=open(dicname,'a')
for i in dictionary:
    fw.write(i+":"+str(dictionary[i])+"\n")
fw.close()
del words  # Hint to reduce memory.
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

# Step 3: Function to generate a training batch for the skip-gram model.
linedata = open(filename, 'r').readlines()
blockmax=(len(linedata)//blockLine)

def generate_shuffleData(block):#
  batch_label=[]
  block=block%(blockmax-1)
  for line in linedata[block*blockLine:(block+1)*blockLine]:#
    sp=line.replace("\n","").split("###")#title,content
    for cw in sp[1].split(" "):
      con=[]
      if(cw not in con):
        con.append(cw)
        for tw in sp[0].split(" "):
          if (cw in dictionary) and (tw in dictionary):
               batch_label .append((dictionary[cw],dictionary[tw]))
  random.shuffle(batch_label)
  return block,batch_label

data_index = 0
blocks,shuffleData = generate_shuffleData(0)
def generate_batch(batchsize):
  global blocks
  global data_index
  global shuffleData
  if(data_index>(len(shuffleData)-batchsize-1)):
      print ("block:"+str(blocks))
      data_index=0
      blocks,shuffleData=generate_shuffleData(blocks+1)
  batch = np.ndarray(shape=(batchsize), dtype=np.int32)
  labels = np.ndarray(shape=(batchsize, 1), dtype=np.int32)
  for i in range(batchsize):
    buffer=shuffleData[data_index]
    batch[i] = buffer[0]
    labels[i,0] = buffer[1]
    data_index +=1
  return batch, labels

batch, labels = generate_batch(12)
for i in range(12):
  print(batch[i], reverse_dictionary[batch[i]],
      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

# Step 4: Build and train a skip-gram model.

unigrams = [ c / vocabulary_size for token, c in count ]



# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
valid_size = 30     # Random set of words to evaluate similarity on.
valid_window = 100  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64    # Number of negative examples to sample.

graph = tf.Graph()

with graph.as_default():

  # Input data.
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

  # Ops and variables pinned to the CPU because of missing GPU implementation
  with tf.device('/cpu:0'):
    # Look up embeddings for inputs.
    # embeddings = tf.Variable(
    #     tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
    # embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    #
    # # Construct the variables for the NCE loss
    # nce_weights = tf.Variable(
    #     tf.truncated_normal([vocabulary_size, embedding_size],
    #                         stddev=1.0 / math.sqrt(embedding_size)))
    # nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


    input_ids = train_inputs
    labels = tf.reshape(train_labels, [batch_size])
    # [vocabulary_size, emb_dim] - input vectors
    input_vectors = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
        name="input_vectors")

    # [vocabulary_size, emb_dim] - output vectors
    output_vectors = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0),
        name="output_vectors")

    # [batch_size, 1] - labels
    labels_matrix = tf.reshape(
        tf.cast(labels,
                dtype=tf.int64),
        [batch_size, 1])

    # Negative sampling.
    sampled_ids, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
        true_classes=labels_matrix,
        num_true=1,
        num_sampled=200,
        unique=True,
        range_max=vocabulary_size,
        distortion=0.75,
        unigrams=unigrams))

    # [batch_size, emb_dim] - Input vectors for center words
    center_vects = tf.nn.embedding_lookup(input_vectors, input_ids)
    # [batch_size, emb_dim] - Output vectors for context words that
    # (center_word, context_word) is in corpus
    context_vects = tf.nn.embedding_lookup(output_vectors, labels)
    # [num_sampled, emb_dim] - vector for sampled words that
    # (center_word, sampled_word) probably isn't in corpus
    sampled_vects = tf.nn.embedding_lookup(output_vectors, sampled_ids)
    # compute logits for pairs of words that are in corpus
    # [batch_size, 1]
    incorpus_logits = tf.reduce_sum(tf.multiply(center_vects, context_vects), 1)
    incorpus_probabilities = tf.nn.sigmoid(incorpus_logits)

    # Sampled logits: [batch_size, num_sampled]
    # We replicate sampled noise labels for all examples in the batch
    # using the matmul.

    sampled_logits = tf.matmul(center_vects,
                               sampled_vects,
                               transpose_b=True)
    outcorpus_probabilities = tf.nn.sigmoid(-sampled_logits)


  # Compute the average NCE loss for the batch.
  # tf.nce_loss automatically draws a new sample of the negative labels each
  # time we evaluate the loss.
  # [batch_size, 1]
  outcorpus_loss_perexample = tf.reduce_sum(tf.log(outcorpus_probabilities), 1)
  loss_perexample = - tf.log(incorpus_probabilities) - outcorpus_loss_perexample

  loss =  tf.reduce_sum(loss_perexample) / batch_size

  # Construct the SGD optimizer using a learning rate of 0.4.
  optimizer = tf.train.GradientDescentOptimizer(.4).minimize(loss)

  # Compute the cosine similarity between minibatch examples and all embeddings.
  norm = tf.sqrt(tf.reduce_sum(tf.square(input_vectors + output_vectors), 1, keep_dims=True))
  normalized_embeddings = (input_vectors + output_vectors) / norm
  valid_embeddings = tf.nn.embedding_lookup(
      normalized_embeddings, valid_dataset)
  similarity = tf.matmul(
      valid_embeddings, normalized_embeddings, transpose_b=True)

  # Add variable initializer.
  init = tf.initialize_all_variables()

# Step 5: Begin training.



with tf.Session(graph=graph) as session:
  # We must initialize all variables before we use them.
  init.run()
  print("Initialized")
  average_loss = 0
  for step in xrange(num_steps):
    batch_inputs, batch_labels = generate_batch(
        batch_size)
    feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

    # We perform one update step by evaluating the optimizer op (including it
    # in the list of returned values for session.run()
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 1000 == 0:
      if step > 0:
        average_loss /= 1000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print("Average loss at step ", step, ": ", average_loss)
      average_loss = 0
      final_embeddings = normalized_embeddings.eval()
      w = open(vecname, 'a')
      for i in final_embeddings[:len(final_embeddings), :].tolist():
        w.write(str(i) + "\n")
      w.close()
      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = "Nearest to %s:" % valid_word
          for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = "%s %s," % (log_str, close_word)
          print(log_str)

# Step 6: Visualize the embeddings.

# def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
#   assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
#   plt.figure(figsize=(18, 18))  #in inches
#   for i, label in enumerate(labels):
#     x, y = low_dim_embs[i,:]
#     plt.scatter(x, y)
#     plt.annotate(label,
#                  xy=(x, y),
#                  xytext=(5, 2),
#                  textcoords='offset points',
#                  ha='right',
#                  va='bottom')
#
#   plt.savefig(filename)
#
# try:
#   from sklearn.manifold import TSNE
#   import matplotlib.pyplot as plt
#
#   tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
#   plot_only = 500
#   low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])
#   labels = [reverse_dictionary[i] for i in xrange(plot_only)]
#   plot_with_labels(low_dim_embs, labels)
#
# except ImportError:
#   print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
