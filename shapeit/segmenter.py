# -*- coding: utf-8 -*-

################################################################################
# Author        : C.Mateis, AIT Austrian Institute of Technology GmbH          #
# Created       : 21.12.2019                                                   #
# Last Modified : 14.01.2020                                                   #
# Description   : Script for decomposing a trace of signal values s at times t #
#                 into a minimum number of lines such that the maximum among   #
#                 the mean squared errors of the regression lines is below a   #
#                 specified threshold error.                                   #
#                 NOTE: An alternative would be using the (more efficient!?)   #
#                       Ramer–Douglas–Peucker algorithm which is successfully  #
#                       used in computer vision to approximate a curve with    #
#                       points (available in OpenCV python library as          #
#                       cv2.approxPolyDP() => import cv2).                     #
#                       The difference is that the Douglas-Peucker algorithm   #
#                       uses the Hausdorff distance and returns a subset of    #
#                       data points of the initial set of data points where    #
#                       two consecutive data points are the margins of a line  #
#                       from the decomposition.                                #
#                       See following link:                                    #
#              - https://en.wikipedia.org/wiki/Ramer–Douglas–Peucker_algorithm #
#                                                                              #
################################################################################

import numpy as np
import sys
import matplotlib.pyplot as plt

from tabulate import tabulate

import random


def computeRegr(aggregate, n):
    """
    Compute the error (mse), the slope and the offset of the linear regression from the terms passed in 'aggregate'
    for a signal slice of length 'n' by using the closed form (i.e. the nromal equation) of the linear regression.

    Parameters
    ----------
    aggregate: 5-tuple of floats
       A tuple of 5 floats containing the terms necessary to compute the regression result in the closed form (normal
       equation): Sx = Sum_i(x[i]), Sy = Sum_i(y[i]), Sxy = Sum_i(x[i]*y[i]), Sxx = Sum_i(x[i]^2), Syy = Sum_i(y[i]^2)
       where x and y are the independent and dependent variables, respectively.
    n: integer
       The number of data points used to compute the terms in the tuple 'aggregate'

    Return
    ------
    A 3-tuple containing the mean squared error (mse), the slope and the offset of the linear regression.

    NOTE
    ----
    See https://en.wikipedia.org/wiki/Simple_linear_regression#Numerical_example
    """
    (Sx, Sy, Sxy, Sxx, Syy) = aggregate
    den = n * Sxx - Sx ** 2

    slope = (n * Sxy - Sx * Sy) / den
    offset = (Sxx * Sy - Sx * Sxy) / den
    err = (1. / n) * (
            Syy - 2 * slope * Sxy - 2 * offset * Sy + (slope ** 2) * Sxx + 2 * slope * offset * Sx) + offset ** 2

    return (err, slope, offset)


def fillRegrTableFrom(errTable, regrTable, x, y, k, mode):
    """
    Fill the regression tables starting from the cell (k,k) either in a row (if mode == 'right') or
    in a column (if mode == 'up') until either the end of the row/column or an already filled cell is reached.
    Return the row/column index of the last cell filled.

    Parameters
    ----------
    errTable: matrix
       A matrix which stores the regression errors for all possible slices of the signal. The cell (i,j) at row i
       and column j stores the regression error of the slice starting at index i and ending at index j in the signal.
       Only the part of the matrix above the main diagonal (i.e. only the cells (i,j) with j >= i) makes sense and is
       used. The regression of a single data point is undefined, thus the cells of the main diagonal are set to np.nan.
    regrTable: matrix
       similar to errTable but storing the pairs (slope,offset) instead of the mse's of the slices regressions
    x: array of float
       The independent variable, i.e. the sampling time points.
    y: array of float
       The dependent variable, i.e. the signal values at the time points given in x.
    k: integer
       The index in the signal giving the first data point of the slices for which the regressions are going to be
       computed. This corresponds to a cell on the first diagonal of the regression tables errTable and regrTable.
       The slices are incrementally extended starting from this data point (either forward, if mode is 'right', or
       backward, if mode is 'up', through the signal).
    mode: string
       A string (possible values: {'right', 'up'}) indicating the direction in which errTable has to be filled.
       If 'right' then the slice is extended incrementally forward, i.e. to the right starting from j.
       If 'up' then the slice is extended incrementally backward, i.e. to the left starting from i.

    Return
    ------
    The index in the signal giving the slice (either forward or backward through the signal starting from index k) for
    which the last regression was computed before the method terminated. If 'mode' is set to 'right', resp. 'up',
    the returned index is either len(x)-1 (i.e. we reached the end of the signal) or the index i-1 *before* the first
    index i encountered along the way for which the regression for [k:i] was already computed in the past coming from
    the bottom (in another method call with mode 'up'), resp. either 0 (i.e. we reached the beginning of the signal) or
    the index i+1 *after* the first index i encountered along the way for which the regression for [i:k] was already
    computed in the past coming from the left (in another method call with mode 'right').
    Note that if we encounter a cell in errTable which was already filled in the past (in another method call) we can
    terminate the method execution because we don't need to (re-)compute the cells beyond that cell.

    NOTE
    ----
    Computing the online regression by simply computing incrementally the terms from the closed form (i.e. normal
    equations) of the linear regression is not recommended since this method becomes numerically unstable for longer
    sequences of data! There are better methods, e.g. Welford's method, which are more stable.
    This aspect needs to be analyzed more thoroughly. Start by looking at the following:
    - Jerome H. Klotz, "UPDATING SIMPLE LINEAR REGRESSION", http://www3.stat.sinica.edu.tw/statistica/oldpdf/A5n124.pdf
    - https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    - https://stats.stackexchange.com/questions/332951/online-algorithm-for-the-mean-square-error
    """
    # initialization:
    errTable[k, k], regrTable[k, k] = np.nan, (np.nan, np.nan)  # the regression of a single data point is undefined
    (Sx, Sy, Sxy, Sxx, Syy) = (x[k], y[k], x[k] * y[k], x[k] ** 2, y[k] ** 2)  # the terms for the normal equation
    (i, j) = (k, k)  # the first cell in errTable and regrTable which is considered in this method call
    n = 1  # the current slice only has 1 data point at the beginning
    # online regression:
    while (((mode == 'right') and (j < len(x) - 1) and (errTable[i, j + 1] is None)) or
           ((mode == 'up') and (i > 0) and (errTable[i - 1, j] is None))):
        if mode == 'right':
            j = j + 1
            (x_add, y_add) = (x[j], y[j])
        elif mode == 'up':
            i = i - 1
            (x_add, y_add) = (x[i], y[i])
        else:
            sys.exit("ERROR: mode '{}' not supported. Execution aborted.".format(mode))

        n += 1
        Sx += x_add
        Sy += y_add
        Sxy += x_add * y_add
        Sxx += x_add ** 2
        Syy += y_add ** 2

        # compute regression:
        (err, slope, offset) = computeRegr((Sx, Sy, Sxy, Sxx, Syy), n)

        # update regression tables:
        errTable[i, j] = err
        regrTable[i, j] = (slope, offset)

    if mode == 'right':
        return j
    elif mode == 'up':
        return i
    else:
        sys.exit("ERROR: mode '{}' not supported. Execution aborted.".format(mode))


def optimalSplit(t, s, i, j, errThreshold, optTable, errTable, regrTable, regrCoverage):
    """
    Try to split the signal s into a minimum number of slices, such that the mean squared errors of the linear
    regressions of the slices are all bellow errThreshold.
    At each point in time given by t[k], the value of the signal is given by s[k].

    Parameters
    ----------
    t: array
       time values
    s: array
       signal values
    i, j: integer
       start, resp. end, index of the slice of s/t; note that j is included and not like in numpy arrays where the
       end index is excluded (we simply use j+1 as the end index of the numpy array)
    errThreshold: float
       maximum regression error (mse) accepted. This is not allowed to be negative!
    optTable: matrix
       a table which stores the optimal split solutions for all slices of the signal as they are computed, i.e.
       optTable[i,j] stores the result returned by optimalSplit(t,s,i,j,...).
       this is useful to avoid optimal split recomputation of a slice [i:j] over and over again every time the slice
       [i:j] is an optimization sub-problem of some other optimization problem
    errTable: matrix
       a table which stores the regression errors/mse's; the cell (i,j) stores the mse for the slice s[i:j]
    regrTable: matrix
       similar to errTable but storing the pairs (slope,offset) instead of the mse's of the slices regressions
    regrCoverage: 2-array of integers of shape=(len(s),2)
       a matrix with len(s) rows and 2 columns. The index of the row i indicates the index of a data point in the
       signal s. The first column (i.e. the column with index 0) indicates the largest slice for which the regression
       was computed along the way starting from index i; it is set to None if no slice starting from i was considered
       so far. The second column (i.e. the column with index 1) indicates the largest slice fro which the regression
       was computed along the way ending at index i; it is set to None if no slice ending at i was considered so far.
       E.g. regrCoverage[i,0] will store the index k of the largest slice starting at i for which the regression was
       computed; note that the regressions of the slices [i:i+1], [i:i+2], ..., [i:k-1] were computed and stored in
       errTable as well, while the regression for the slice [i:k] was computed incrementally moving *forwards* in the
       signal starting from [i:i].
       E.g. regrCoverage[i,1] will store the index k of the largest slice ending at i for which the regression was
       computed; note that the regressions of the slices [k+1:i], [k+2:i], ..., [i-1:i] were computed and stored in
       errTable as well, while the regression for the slice [k:i] was computed incrementally moving *backwards* in the
       signal starting from [i:i].

    Return
    ------
    A tuple (optErr, optIndexSplit) where optIndexSplit is the list of the optimal indices split in s/t and optErr is the
    maximum mean squared error among all linear regressions of the splits. Note that s/t are indexed starting from 0 until
    len(s)-1. The first (i.e. 0) and last (i.e. len(s)-1) indices are excluded from optIndexSplit. E.g. optIndexSplit=[50,75]
    indicates that the signal decomposition consists of 2 split points at indices 50 and 75, meaning that the signal
    decomposition consists of the three linear regression lines of the slices [0:50], [50:75] and [75:len(s)-1].
    Special cases:
    1. If the linear regression of the whole signal given in input yields an error 'err' which is smaller than
    errThreshold then the function returns (err, []).
    2. If errThreshold == 0 then the optIndexSplit will be every single point in the worst case (unless there are
       portions in the signal which are trully linear).
    """

    if optTable[i][j] is None:
        if regrCoverage[i][0] is None:
            # the required slices starting at index i were not considered for regression before;
            # compute these regressions now
            regrCoverage[i][0] = fillRegrTableFrom(errTable, regrTable, t, s, i, 'right')

        err = errTable[i, j]
        if (err <= errThreshold):
            # we are done; we can fit the slice with a single regression line, no need for splits
            optErr = err
            optIndexSplit = []
        else:
            # the regression error of a single line is above the threshold error; splits are necessary!
            # we solve the optimization problem by semi-dynamic programming:
            #    - we iterate over all possible split index candidates k \in [i+1, j-1]
            #    - at each iteration k we solve the optimization sub-problem for the prefix slice [i:k] and we compute the
            #      error for the split set {splits([i:k]), k} as max(err([i:k]), err([k:j])), where splits([i:k]) and
            #      err([i:k]) are the optimal split results for the slice [i:k] returned by the optimization sub-problem
            #      and err([k:j]) is the error of a *single* regression line for the remaining slice [k:j] (the suffix slice)
            #    - at each iteration k we update the optimal solution for [i:j] if it is better than the last optimal solution;
            #      better means (1) it does not exceed the errThreshold, and (2.1) it has a lower number of splits (i.e. lines)
            #      or (2.2) it has the same number of splits (i.e. lines) but the error is lower

            if regrCoverage[j][1] is None:
                # the required slices ending at index j were not considered for regression before;
                # compute these regressions now
                regrCoverage[j][1] = fillRegrTableFrom(errTable, regrTable, t, s, j, 'up')

            # initialization with the worst case situation, i.e. maximum number of lines and regression error for each line equal to 0
            optErr = 0
            optIndexSplit = list(range(i + 1, j, 1))
            for k in range(i + 1, j):
                errRight = errTable[k, j]
                # try to generate a candidate only if errRight <= errThreshold, otherwise (i.e. errRight > errThreshold) we know
                #   already that this candidate would violate the error threshold constraint because the following would hold:
                #                err >= errRight > errThreshold
                #   since err = max(?,errRight) >= errRight:
                if errRight <= errThreshold:
                    (optErrLeft, optIndexSplitLeft) = optimalSplit(t, s, i, k, errThreshold, optTable, errTable,
                                                                   regrTable, regrCoverage)
                    err = max(optErrLeft, errRight)
                    indexSplit = optIndexSplitLeft + [k]
                    if ((len(indexSplit) < len(
                            optIndexSplit)) or  # the current split set has less lines than the last one, hence better
                            (len(indexSplit) == len(optIndexSplit)) and (
                                    err < optErr)):  # the current split set has the same number of lines but a smaller error than the last one, hence better
                        # update the optimal split set with the current one which is better:
                        optErr = err
                        optIndexSplit = indexSplit

        # save the result to 'optTable' for later re-use
        optTable[i][j] = (optErr, optIndexSplit)
    else:
        # take the result from 'optTable' instead of re-computing it
        (optErr, optIndexSplit) = optTable[i][j]

    return (optErr, optIndexSplit)


def compute_optimal_splits(t, s, errThreshold, debug=False):
    """
    Given a trace 's' of signal values and the corresponding timestamps vector 't', find the best split/segmentation of 's',
    i.e. the split with the minimum number of lines, such that the maximum among the linear regression errors of the splits
    does not exceed the specified error threshold 'errThreshold'.

    Parameters
    ----------
    t: array of floats
       a vector containing the sampling points in time (in seconds)
    s: array of floats
       a vector containing the signal values corresponing to the points in time stored in t.
       must have the same size as t
    errThreshold: float
       the maximum value we accept as error for the trace split/segmentation. Since we define the error of a split as the
       maximum linear regression error among the linear regression errors of the individual segments, errThreshold is actually
       the maximum linear regression error we accept for the linear regression of each segment.
       must be positive

    Return
    ------
    A table which stores in the rows the details related to the splits (one row for one split):
    - nr => number of the current split/segment in the (ordered) segmentation of the signal according to the identified
            optimal solution
    - index_start => the index in the signal s where the current split/segment starts
    - index_end => the index in the signal s where the current split/segment ends
    - slope => the slope of the regression line of the current split/segment
    - offset => the ofset of the regression line of the current split/segment
    - mse => the mean squared error of the linear regression of the current split/segment
    - duration => the duration in seconds of the current split/segment
    """
    assert ((len(s) > 1) and (errThreshold >= 0.))
    # tables initialization (look at the documentation of the methods above to understand the meaning of these tables):
    optTable = np.tile(None, (len(s), len(s)))
    errTable = np.tile(None, (len(s), len(s)))
    regrTable = np.tile(None, (len(s), len(s)))
    regrCoverage = np.tile(None, (len(s), 2))

    _, optIndexSplit = optimalSplit(t, s, 0, len(s) - 1, errThreshold, optTable, errTable, regrTable, regrCoverage)

    # this is to check how the table errTable is filled:
    if debug:
        # print(tabulate(errTable == None, headers='keys', showindex=True, tablefmt='psql'))
        print(tabulate(errTable == None, showindex=False))

    resultTable = np.zeros((len(optIndexSplit) + 1, 7))
    index_start = 0
    for i in range(len(optIndexSplit) + 1):
        if i < len(optIndexSplit):
            index_end = optIndexSplit[i]
        else:
            index_end = len(s) - 1
        (slope, offset) = regrTable[index_start, index_end]
        mse = errTable[index_start, index_end]
        duration = t[index_end] - t[index_start]
        resultTable[i] = np.array([i + 1, index_start, index_end, slope, offset, mse, duration])
        index_start = index_end

    return resultTable


def plot_splits(t, s, resultTable, plotLegend=True):
    """
    Plot the original time series signal (t,s) and a segmentation of it computed with compute_optimal_splits(t,s,errTHreshold).
    The passed parameter 'resultTable' must be the return of an invokation of compute_optimal_splits().
    """
    fig = plt.figure(figsize=(12, 7))
    plt.plot(t, s, label="signal")

    for k in range(resultTable.shape[0]):
        i, j = resultTable[k, 1:3].astype(int)
        slope, offset, mse = resultTable[k, 3:6]
        x0, y0 = t[i], slope * t[i] + offset
        x1, y1 = t[j], slope * t[j] + offset
        plt.plot([x0, x1], [y0, y1], label="line {} (mse={})".format(k + 1, mse))

    if plotLegend:
        plt.legend()

    return fig