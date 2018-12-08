[Jananese](README_ja.md)/ English

# Chainer Example

## Summary

The first step of Chainer.

## Usage

    ruby makedata.rb  # make dataset for training
    python train.py   # training
    python test.py    # check the obtained model
    python export.py  # export model for C++
    make              # build a C++ test code
    ./a.out           # import model and test it

## Dataset

* makedata.rb: identify convex upword or downward
* evenodd.rb : identify even or odd number of bits
