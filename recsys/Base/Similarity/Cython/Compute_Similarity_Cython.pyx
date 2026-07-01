"""
Created on 23/10/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=True
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

"""
Determine the operative system. The interface of numpy returns a different type for argsort under windows and linux

http://docs.cython.org/en/latest/src/userguide/language_basics.html#conditional-compilation
"""
IF UNAME_SYSNAME == "linux":
    DEF LONG_t = "long"
ELIF  UNAME_SYSNAME == "Windows":
    DEF LONG_t = "long long"
ELSE:
    DEF LONG_t = "long long"



import time, sys
import cython
import numpy as np
cimport numpy as np

from cpython.array cimport array, clone

from libc.math cimport sqrt




import scipy.sparse as sps
from recsys.Base.Recommender_utils import check_matrix

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class Compute_Similarity_Cython:

    cdef int TopK
    cdef long n_columns, n_rows

    cdef double[:] this_item_weights
    cdef int[:] this_item_weights_mask, this_item_weights_id
    cdef int this_item_weights_counter

    cdef int[:] user_to_item_row_ptr, user_to_item_cols
    cdef int[:] item_to_user_rows, item_to_user_col_ptr
    cdef double[:] user_to_item_data, item_to_user_data
    cdef double[:] sumOfSquared, sumOfSquared_to_1_minus_alpha, sumOfSquared_to_alpha
    cdef int shrink, normalize, adjusted_cosine, pearson_correlation, tanimoto_coefficient, asymmetric_cosine, dice_coefficient, tversky_coefficient
    cdef float asymmetric_alpha, tversky_alpha, tversky_beta

    cdef int use_row_weights
    cdef double[:] row_weights

    cdef double[:,:] W_dense

    def __init__(self, dataMatrix, topK = 100, shrink=0, normalize = True,
                 asymmetric_alpha = 0.5, tversky_alpha = 1.0, tversky_beta = 1.0,
                 similarity = "cosine", row_weights = None):
        """
        Computes the cosine similarity on the columns of dataMatrix
        If it is computed on URM=|users|x|items|, pass the URM as is.
        If it is computed on ICM=|items|x|features|, pass the ICM transposed.
        :param dataMatrix:
        :param topK:
        :param shrink:
        :param normalize:           If True divide the dot product by the product of the norms
        :param row_weights:         Multiply the values in each row by a specified value. Array
        :param asymmetric_alpha     Coefficient alpha for the asymmetric cosine
        :param similarity:  "cosine"        computes Cosine similarity
                            "adjusted"      computes Adjusted Cosine, removing the average of the users
                            "asymmetric"    computes Asymmetric Cosine
                            "pearson"       computes Pearson Correlation, removing the average of the items
                            "jaccard"       computes Jaccard similarity for binary interactions using Tanimoto
                            "dice"          computes Dice similarity for binary interactions
                            "tversky"       computes Tversky similarity for binary interactions
                            "tanimoto"      computes Tanimoto coefficient for binary interactions

        """
        """
        Asymmetric Cosine as described in: 
        Aiolli, F. (2013, October). Efficient top-n recommendation for very large scale binary rated datasets. In Proceedings of the 7th ACM conference on Recommender systems (pp. 273-280). ACM.
        
        """

        super(Compute_Similarity_Cython, self).__init__()

        self.n_columns = dataMatrix.shape[1]
        self.n_rows = dataMatrix.shape[0]
        self.shrink = shrink
        self.normalize = normalize
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.adjusted_cosine = False
        self.asymmetric_cosine = False
        self.pearson_correlation = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "adjusted":
            self.adjusted_cosine = True
        elif similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "pearson":
            self.pearson_correlation = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            # Tanimoto has a specific kind of normalization
            self.normalize = False

        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False

        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False

        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Cosine_Similarity: value for parameter 'mode' not recognized."
                             " Allowed values are: 'cosine', 'pearson', 'adjusted', 'asymmetric', 'jaccard', 'tanimoto',"
                             "dice, tversky."
                             " Passed value was '{}'".format(similarity))


        self.TopK = min(topK, self.n_columns)
        self.this_item_weights = np.zeros(self.n_columns, dtype=np.float64)
        self.this_item_weights_id = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_mask = np.zeros(self.n_columns, dtype=np.int32)
        self.this_item_weights_counter = 0

        # Copy data to avoid altering the original object
        dataMatrix = dataMatrix.copy()





        if self.adjusted_cosine:
            dataMatrix = self.applyAdjustedCosine(dataMatrix)
        elif self.pearson_correlation:
            dataMatrix = self.applyPearsonCorrelation(dataMatrix)
        elif self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient:
            dataMatrix = self.useOnlyBooleanInteractions(dataMatrix)



        # Compute sum of squared values to be used in normalization
        self.sumOfSquared = np.array(dataMatrix.power(2).sum(axis=0), dtype=np.float64).ravel()

        # Tanimoto does not require the square root to be applied
        if not (self.tanimoto_coefficient or self.dice_coefficient or self.tversky_coefficient):
            self.sumOfSquared = np.sqrt(self.sumOfSquared)

        if self.asymmetric_cosine:
            self.sumOfSquared_to_1_minus_alpha = np.power(self.sumOfSquared, 2 * (1 - self.asymmetric_alpha))
            self.sumOfSquared_to_alpha = np.power(self.sumOfSquared, 2 * self.asymmetric_alpha)


        # Apply weight after sumOfSquared has been computed but before the matrix is
        # split in its inner data structures
        self.use_row_weights = False

        if row_weights is not None:

            if dataMatrix.shape[0] != len(row_weights):
                raise ValueError("Cosine_Similarity: provided row_weights and dataMatrix have different number of rows."
                                 "Row_weights has {} rows, dataMatrix has {}.".format(len(row_weights), dataMatrix.shape[0]))


            self.use_row_weights = True
            self.row_weights = np.array(row_weights, dtype=np.float64)





        dataMatrix = check_matrix(dataMatrix, 'csr')

        self.user_to_item_row_ptr = dataMatrix.indptr
        self.user_to_item_cols = dataMatrix.indices
        self.user_to_item_data = np.array(dataMatrix.data, dtype=np.float64)

        dataMatrix = check_matrix(dataMatrix, 'csc')
        self.item_to_user_rows = dataMatrix.indices
        self.item_to_user_col_ptr = dataMatrix.indptr
        self.item_to_user_data = np.array(dataMatrix.data, dtype=np.float64)




        if self.TopK == 0:
            self.W_dense = np.zeros((self.n_columns,self.n_columns))





    cdef useOnlyBooleanInteractions(self, dataMatrix):
        """
        Set to 1 all data points
        :return:
        """

        cdef long index

        for index in range(len(dataMatrix.data)):
            dataMatrix.data[index] = 1

        return dataMatrix



    cdef applyPearsonCorrelation(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding column
        :return:
        """

        cdef double[:] sumPerCol
        cdef int[:] interactionsPerCol
        cdef long colIndex, innerIndex, start_pos, end_pos
        cdef double colAverage


        dataMatrix = check_matrix(dataMatrix, 'csc')


        sumPerCol = np.array(dataMatrix.sum(axis=0), dtype=np.float64).ravel()
        interactionsPerCol = np.diff(dataMatrix.indptr)


        #Remove for every row the corresponding average
        for colIndex in range(self.n_columns):

            if interactionsPerCol[colIndex]>0:

                colAverage = sumPerCol[colIndex] / interactionsPerCol[colIndex]

                start_pos = dataMatrix.indptr[colIndex]
                end_pos = dataMatrix.indptr[colIndex+1]

                innerIndex = start_pos

                while innerIndex < end_pos:

                    dataMatrix.data[innerIndex] -= colAverage
                    innerIndex+=1


        return dataMatrix



    cdef applyAdjustedCosine(self, dataMatrix):
        """
        Remove from every data point the average for the corresponding row
        :return:
        """

        cdef double[:] sumPerRow
        cdef int[:] interactionsPerRow
        cdef long rowIndex, innerIndex, start_pos, end_pos
        cdef double rowAverage

        dataMatrix = check_matrix(dataMatrix, 'csr')

        sumPerRow = np.array(dataMatrix.sum(axis=1), dtype=np.float64).ravel()
        interactionsPerRow = np.diff(dataMatrix.indptr)


        #Remove for every row the corresponding average
        for rowIndex in range(self.n_rows):

            if interactionsPerRow[rowIndex]>0:

                rowAverage = sumPerRow[rowIndex] / interactionsPerRow[rowIndex]

                start_pos = dataMatrix.indptr[rowIndex]
                end_pos = dataMatrix.indptr[rowIndex+1]

                innerIndex = start_pos

                while innerIndex < end_pos:

                    dataMatrix.data[innerIndex] -= rowAverage
                    innerIndex+=1


        return dataMatrix





    cdef int[:] getUsersThatRatedItem(self, long item_id):
        return self.item_to_user_rows[self.item_to_user_col_ptr[item_id]:self.item_to_user_col_ptr[item_id+1]]

    cdef int[:] getItemsRatedByUser(self, long user_id):
        return self.user_to_item_cols[self.user_to_item_row_ptr[user_id]:self.user_to_item_row_ptr[user_id+1]]




    cdef computeItemSimilarities(self, long item_id_input):
        """
        For every item the cosine similarity against other items depends on whether they have users in common. The more
        common users the higher the similarity.
        
        The basic implementation is:
        - Select the first item
        - Loop through all other items
        -- Given the two items, get the users they have in common
        -- Update the similarity for all common users
        
        That is VERY slow due to the common user part, in which a long data structure is looped multiple times.
        
        A better way is to use the data structure in a different way skipping the search part, getting directly the
        information we need.
        
        The implementation here used is:
        - Select the first item
        - Initialize a zero valued array for the similarities
        - Get the users who rated the first item
        - Loop through the users
        -- Given a user, get the items he rated (second item)
        -- Update the similarity of the items he rated
        
        
        """

        # Create template used to initialize an array with zeros
        # Much faster than np.zeros(self.n_columns)
        #cdef array[double] template_zero = array('d')
        #cdef array[double] result = clone(template_zero, self.n_columns, zero=True)


        cdef long user_index, user_id, item_index, item_id, item_id_second

        cdef int[:] users_that_rated_item = self.getUsersThatRatedItem(item_id_input)
        cdef int[:] items_rated_by_user

        cdef double rating_item_input, rating_item_second, row_weight

        # Clean previous item
        for item_index in range(self.this_item_weights_counter):
            item_id = self.this_item_weights_id[item_index]
            self.this_item_weights_mask[item_id] = False
            self.this_item_weights[item_id] = 0.0

        self.this_item_weights_counter = 0



        # Get users that rated the items
        for user_index in range(len(users_that_rated_item)):

            user_id = users_that_rated_item[user_index]
            rating_item_input = self.item_to_user_data[self.item_to_user_col_ptr[item_id_input]+user_index]

            if self.use_row_weights:
                row_weight = self.row_weights[user_id]
            else:
                row_weight = 1.0

            # Get all items rated by that user
            items_rated_by_user = self.getItemsRatedByUser(user_id)

            for item_index in range(len(items_rated_by_user)):

                item_id_second = items_rated_by_user[item_index]

                # Do not compute the similarity on the diagonal
                if item_id_second != item_id_input:
                    # Increment similairty
                    rating_item_second = self.user_to_item_data[self.user_to_item_row_ptr[user_id]+item_index]

                    self.this_item_weights[item_id_second] += rating_item_input*rating_item_second*row_weight


                    # Update global data structure
                    if not self.this_item_weights_mask[item_id_second]:

                        self.this_item_weights_mask[item_id_second] = True
                        self.this_item_weights_id[self.this_item_weights_counter] = item_id_second
                        self.this_item_weights_counter += 1




    def compute_similarity(self, start_col=None, end_col=None):
        """
        Compute the similarity for the given dataset
        :param self:
        :param start_col: column to begin with
        :param end_col: column to stop before, end_col is excluded
        :return:
        """

        cdef int print_block_size = 500

        cdef int itemIndex, innerItemIndex, item_id, local_topK
        cdef long long topKItemIndex

        cdef long long[:] top_k_idx

        # Declare numpy data type to use vetor indexing and simplify the topK selection code
        cdef np.ndarray[LONG_t, ndim=1] top_k_partition, top_k_partition_sorting
        cdef np.ndarray[np.float64_t, ndim=1] this_item_weights_np = np.zeros(self.n_columns, dtype=np.float64)
        #cdef double[:] this_item_weights

        cdef long processedItems = 0

        # Data structure to incrementally build sparse matrix
        # Preinitialize max possible length
        cdef double[:] values = np.zeros((self.n_columns*self.TopK))
        cdef int[:] rows = np.zeros((self.n_columns*self.TopK,), dtype=np.int32)
        cdef int[:] cols = np.zeros((self.n_columns*self.TopK,), dtype=np.int32)
        cdef long sparse_data_pointer = 0

        cdef int start_col_local = 0, end_col_local = self.n_columns

        cdef array[double] template_zero = array('d')



        if start_col is not None and start_col>0 and start_col<self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col>start_col_local and end_col<self.n_columns:
            end_col_local = end_col






        start_time = time.time()
        last_print_time = start_time

        itemIndex = start_col_local

        # Compute all similarities for each item
        while itemIndex < end_col_local:

            processedItems += 1

            # Computed similarities go in self.this_item_weights
            self.computeItemSimilarities(itemIndex)


            # Apply normalization and shrinkage only to non-zero candidates.
            if self.normalize:
                for innerItemIndex in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[innerItemIndex]

                    if self.asymmetric_cosine:
                        self.this_item_weights[item_id] /= self.sumOfSquared_to_alpha[itemIndex] * self.sumOfSquared_to_1_minus_alpha[item_id]\
                                                         + self.shrink + 1e-6

                    else:
                        self.this_item_weights[item_id] /= self.sumOfSquared[itemIndex] * self.sumOfSquared[item_id]\
                                                         + self.shrink + 1e-6

            # Apply the specific denominator for Tanimoto
            elif self.tanimoto_coefficient:
                for innerItemIndex in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[innerItemIndex]
                    self.this_item_weights[item_id] /= self.sumOfSquared[itemIndex] + self.sumOfSquared[item_id] -\
                                                     self.this_item_weights[item_id] + self.shrink + 1e-6

            elif self.dice_coefficient:
                for innerItemIndex in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[innerItemIndex]
                    self.this_item_weights[item_id] /= self.sumOfSquared[itemIndex] + self.sumOfSquared[item_id] +\
                                                     self.shrink + 1e-6

            elif self.tversky_coefficient:
                for innerItemIndex in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[innerItemIndex]
                    self.this_item_weights[item_id] /= self.this_item_weights[item_id] + \
                                                      (self.sumOfSquared[itemIndex]-self.this_item_weights[item_id])*self.tversky_alpha + \
                                                      (self.sumOfSquared[item_id]-self.this_item_weights[item_id])*self.tversky_beta +\
                                                      self.shrink + 1e-6

            elif self.shrink != 0:
                for innerItemIndex in range(self.this_item_weights_counter):
                    item_id = self.this_item_weights_id[innerItemIndex]
                    self.this_item_weights[item_id] /= self.shrink


            if self.TopK == 0:

                for innerItemIndex in range(self.n_columns):
                    self.W_dense[innerItemIndex,itemIndex] = self.this_item_weights[innerItemIndex]

            else:

                # Sort indices and select TopK
                # Using numpy implies some overhead, unfortunately the plain C qsort function is even slower
                #top_k_idx = np.argsort(this_item_weights) [-self.TopK:]

                # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
                # because we avoid sorting elements we already know we don't care about
                # - Partition the data to extract the set of TopK items, this set is unsorted
                # - Sort only the TopK items, discarding the rest
                # - Get the original item index
                #



                local_topK = min([self.TopK, self.this_item_weights_counter])

                if local_topK > 0:

                    if self.adjusted_cosine or self.pearson_correlation or self.use_row_weights:
                        for innerItemIndex in range(self.n_columns):
                            this_item_weights_np[innerItemIndex] = 0.0

                        for innerItemIndex in range(self.this_item_weights_counter):
                            item_id = self.this_item_weights_id[innerItemIndex]
                            this_item_weights_np[innerItemIndex] = - self.this_item_weights[item_id]

                        # Get the unordered set of topK items
                        top_k_partition = np.argpartition(this_item_weights_np, local_topK-1)[0:local_topK]

                    else:
                        # Add weights in the same ordering as the self.this_item_weights_id data structure
                        for innerItemIndex in range(self.this_item_weights_counter):
                            item_id = self.this_item_weights_id[innerItemIndex]
                            this_item_weights_np[innerItemIndex] = - self.this_item_weights[item_id]

                        # Get the unordered set of topK items
                        top_k_partition = np.argpartition(this_item_weights_np[0:self.this_item_weights_counter], local_topK-1)[0:local_topK]

                    # Sort only the elements in the partition
                    top_k_partition_sorting = np.argsort(this_item_weights_np[top_k_partition])
                    # Get original index
                    top_k_idx = top_k_partition[top_k_partition_sorting]



                    # Incrementally build sparse matrix, do not add zeros
                    for innerItemIndex in range(len(top_k_idx)):

                        topKItemIndex = top_k_idx[innerItemIndex]

                        item_id = self.this_item_weights_id[topKItemIndex]

                        if self.this_item_weights[item_id] != 0.0:

                            values[sparse_data_pointer] = self.this_item_weights[item_id]
                            rows[sparse_data_pointer] = item_id
                            cols[sparse_data_pointer] = itemIndex

                            sparse_data_pointer += 1


            itemIndex += 1


            if processedItems % print_block_size==0 or processedItems==end_col_local:

                current_time = time.time()

                # Set block size to the number of items necessary in order to print every 30 seconds
                if current_time - start_time != 0:
                    itemPerSec = processedItems/(current_time - start_time)
                else:
                    itemPerSec = 1

                print_block_size = int(itemPerSec*30)

                if current_time - last_print_time > 30  or processedItems==end_col_local:

                    print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                        processedItems, processedItems*1.0/(end_col_local-start_col_local)*100, itemPerSec, (time.time()-start_time) / 60))

                    last_print_time = current_time

                    sys.stdout.flush()
                    sys.stderr.flush()

        # End while on columns


        if self.TopK == 0:

            return np.array(self.W_dense)

        else:

            values = np.array(values[0:sparse_data_pointer])
            rows = np.array(rows[0:sparse_data_pointer])
            cols = np.array(cols[0:sparse_data_pointer])

            t = -time.time()
            print("Building sparse matrix")
            W_sparse = sps.csr_matrix((values, (rows, cols)),
                                    shape=(self.n_columns, self.n_columns),
                                    dtype=np.float32)
            t += time.time()
            print("Building sparse matrix took %.1f minutes" % (t / 60))

            return W_sparse


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
@cython.overflowcheck(False)
cdef class Compute_Similarity_Cached_Cython:

    cdef int TopK
    cdef long n_columns

    cdef int[:] raw_similarity_indices, raw_similarity_indptr
    cdef double[:] raw_similarity_data
    cdef double[:] sumOfSquared, sumOfSquared_to_1_minus_alpha, sumOfSquared_to_alpha
    cdef int normalize, tanimoto_coefficient, asymmetric_cosine, dice_coefficient, tversky_coefficient
    cdef double shrink, asymmetric_alpha, tversky_alpha, tversky_beta
    cdef int max_col_nnz

    def __init__(self, cached_data, topK=100, shrink=0, normalize=True,
                 asymmetric_alpha=0.5, tversky_alpha=1.0, tversky_beta=1.0,
                 similarity="cosine"):

        cdef object raw_similarity
        cdef np.ndarray[np.int32_t, ndim=1] raw_similarity_indices_np
        cdef np.ndarray[np.int32_t, ndim=1] raw_similarity_indptr_np
        cdef np.ndarray[np.float64_t, ndim=1] raw_similarity_data_np
        cdef np.ndarray[np.float64_t, ndim=1] sumOfSquared_np

        super(Compute_Similarity_Cached_Cython, self).__init__()

        self.n_columns = cached_data.n_columns
        self.TopK = min(topK, self.n_columns)
        self.shrink = shrink
        self.normalize = normalize
        self.asymmetric_alpha = asymmetric_alpha
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta

        self.asymmetric_cosine = False
        self.tanimoto_coefficient = False
        self.dice_coefficient = False
        self.tversky_coefficient = False

        if similarity == "asymmetric":
            self.asymmetric_cosine = True
        elif similarity == "jaccard" or similarity == "tanimoto":
            self.tanimoto_coefficient = True
            self.normalize = False
        elif similarity == "dice":
            self.dice_coefficient = True
            self.normalize = False
        elif similarity == "tversky":
            self.tversky_coefficient = True
            self.normalize = False
        elif similarity == "cosine":
            pass
        else:
            raise ValueError("Compute_Similarity_Cached_Cython: value for parameter 'similarity' not recognized."
                             " Allowed values are: 'cosine', 'asymmetric', 'jaccard', 'tanimoto', dice, tversky."
                             " Passed value was '{}'".format(similarity))

        raw_similarity = check_matrix(cached_data.raw_similarity, "csc", dtype=np.float64)
        raw_similarity.sort_indices()

        raw_similarity_indices_np = np.asarray(raw_similarity.indices, dtype=np.int32)
        raw_similarity_indptr_np = np.asarray(raw_similarity.indptr, dtype=np.int32)
        raw_similarity_data_np = np.asarray(raw_similarity.data, dtype=np.float64)

        self.raw_similarity_indices = raw_similarity_indices_np
        self.raw_similarity_indptr = raw_similarity_indptr_np
        self.raw_similarity_data = raw_similarity_data_np

        sumOfSquared_np = np.asarray(cached_data.sum_of_squared, dtype=np.float64)
        self.sumOfSquared = sumOfSquared_np

        if self.asymmetric_cosine:
            self.sumOfSquared_to_1_minus_alpha = np.power(sumOfSquared_np, 2 * (1 - self.asymmetric_alpha))
            self.sumOfSquared_to_alpha = np.power(sumOfSquared_np, 2 * self.asymmetric_alpha)

        if self.n_columns > 0:
            self.max_col_nnz = int(np.max(np.diff(raw_similarity_indptr_np)))
        else:
            self.max_col_nnz = 0

    def compute_similarity(self, start_col=None, end_col=None):
        """
        Score a cached raw column-column co-occurrence matrix.
        The raw cache is CSC, therefore each output column can be normalized and
        top-K filtered without recomputing dataMatrix.T.dot(dataMatrix).
        """

        cdef int print_block_size = 500

        cdef int column_index, row_index, position, start_position, end_position
        cdef int candidate_count, candidate_index, local_topK
        cdef long long topKCandidateIndex

        cdef long processedItems = 0
        cdef long sparse_data_pointer = 0
        cdef long max_output_size

        cdef int start_col_local = 0
        cdef int end_col_local = self.n_columns

        cdef double raw_weight, denominator, similarity_value

        cdef long long[:] top_k_idx
        cdef np.ndarray[LONG_t, ndim=1] top_k_partition, top_k_partition_sorting

        cdef np.ndarray[np.float64_t, ndim=1] candidate_weights_np
        cdef np.ndarray[np.int32_t, ndim=1] candidate_rows_np

        cdef double[:] values
        cdef int[:] rows, cols
        cdef np.ndarray[np.float64_t, ndim=1] values_np
        cdef np.ndarray[np.int32_t, ndim=1] rows_np, cols_np
        cdef object W_sparse

        if self.TopK == 0:
            return None

        if self.max_col_nnz == 0:
            return sps.csr_matrix((self.n_columns, self.n_columns), dtype=np.float32)

        if start_col is not None and start_col > 0 and start_col < self.n_columns:
            start_col_local = start_col

        if end_col is not None and end_col > start_col_local and end_col < self.n_columns:
            end_col_local = end_col

        candidate_weights_np = np.zeros(self.max_col_nnz, dtype=np.float64)
        candidate_rows_np = np.zeros(self.max_col_nnz, dtype=np.int32)

        max_output_size = self.n_columns * self.TopK
        values = np.zeros(max_output_size, dtype=np.float64)
        rows = np.zeros(max_output_size, dtype=np.int32)
        cols = np.zeros(max_output_size, dtype=np.int32)

        start_time = time.time()
        last_print_time = start_time

        column_index = start_col_local

        while column_index < end_col_local:

            processedItems += 1

            start_position = self.raw_similarity_indptr[column_index]
            end_position = self.raw_similarity_indptr[column_index + 1]
            candidate_count = 0

            for position in range(start_position, end_position):

                raw_weight = self.raw_similarity_data[position]

                if raw_weight != 0.0:

                    row_index = self.raw_similarity_indices[position]
                    similarity_value = raw_weight

                    if self.normalize:
                        if self.asymmetric_cosine:
                            denominator = self.sumOfSquared_to_alpha[column_index] * self.sumOfSquared_to_1_minus_alpha[row_index] + \
                                          self.shrink + 1e-6
                        else:
                            denominator = self.sumOfSquared[column_index] * self.sumOfSquared[row_index] + self.shrink + 1e-6

                        similarity_value /= denominator

                    elif self.tanimoto_coefficient:
                        denominator = self.sumOfSquared[column_index] + self.sumOfSquared[row_index] - raw_weight + \
                                      self.shrink + 1e-6
                        similarity_value /= denominator

                    elif self.dice_coefficient:
                        denominator = self.sumOfSquared[column_index] + self.sumOfSquared[row_index] + self.shrink + 1e-6
                        similarity_value /= denominator

                    elif self.tversky_coefficient:
                        denominator = raw_weight + \
                                      (self.sumOfSquared[column_index] - raw_weight) * self.tversky_alpha + \
                                      (self.sumOfSquared[row_index] - raw_weight) * self.tversky_beta + \
                                      self.shrink + 1e-6
                        similarity_value /= denominator

                    elif self.shrink != 0:
                        similarity_value /= self.shrink

                    if similarity_value != 0.0:
                        candidate_weights_np[candidate_count] = similarity_value
                        candidate_rows_np[candidate_count] = row_index
                        candidate_count += 1

            local_topK = min(self.TopK, candidate_count)

            if local_topK > 0:

                if local_topK < candidate_count:
                    top_k_partition = np.argpartition(-candidate_weights_np[0:candidate_count], local_topK - 1)[0:local_topK]
                    top_k_partition_sorting = np.argsort(-candidate_weights_np[top_k_partition])
                    top_k_idx = top_k_partition[top_k_partition_sorting]
                else:
                    top_k_idx = np.argsort(-candidate_weights_np[0:candidate_count])

                for candidate_index in range(len(top_k_idx)):

                    topKCandidateIndex = top_k_idx[candidate_index]

                    values[sparse_data_pointer] = candidate_weights_np[topKCandidateIndex]
                    rows[sparse_data_pointer] = candidate_rows_np[topKCandidateIndex]
                    cols[sparse_data_pointer] = column_index

                    sparse_data_pointer += 1

            column_index += 1

            if processedItems % print_block_size == 0 or processedItems == end_col_local:

                current_time = time.time()

                if current_time - start_time != 0:
                    itemPerSec = processedItems / (current_time - start_time)
                else:
                    itemPerSec = 1

                print_block_size = int(itemPerSec * 30)

                if current_time - last_print_time > 30 or processedItems == end_col_local:

                    print("Similarity column {} ( {:2.0f} % ), {:.2f} column/sec, elapsed time {:.2f} min".format(
                        processedItems, processedItems * 1.0 / (end_col_local - start_col_local) * 100,
                        itemPerSec, (time.time() - start_time) / 60))

                    last_print_time = current_time

                    sys.stdout.flush()
                    sys.stderr.flush()

        values_np = np.array(values[0:sparse_data_pointer])
        rows_np = np.array(rows[0:sparse_data_pointer])
        cols_np = np.array(cols[0:sparse_data_pointer])

        t = -time.time()
        print("Building sparse matrix")
        W_sparse = sps.csr_matrix((values_np, (rows_np, cols_np)),
                                  shape=(self.n_columns, self.n_columns),
                                  dtype=np.float32)
        t += time.time()
        print("Building sparse matrix took %.1f minutes" % (t / 60))

        return W_sparse
