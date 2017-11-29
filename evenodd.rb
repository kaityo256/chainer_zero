def makedata(s,n,l,filename)
  puts filename
  f = open(filename,"w")
  num = 0
  while num < n
    str = Array.new(10){ rand<0.5? 1:0}.join(",")
    if (str.count("1") % 2 == s)
      f.puts str
      num = num + 1
    end
  end
end

TRAIN_DATA = 1000
TEST_DATA = 100
L = 10
makedata(1,TRAIN_DATA,L,"on.txt")
makedata(0,TRAIN_DATA,L,"off.txt")

makedata(1,TEST_DATA,L,"on_test.txt")
makedata(0,TEST_DATA,L,"off_test.txt")
