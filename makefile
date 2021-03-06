all: a.out

a.out: model.cpp model.hpp
	g++ -std=c++11 $< -o $@

clean:
	rm -f a.out

test: a.out
	ruby makedata.rb
	python train.py
	python export.py
	./a.out

clear:
	rm -f a.out *.txt test.model test.dat *.pyc
	rm -rf result
