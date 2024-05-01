https://www.image-net.org/

mkdir train
mkdir val
tar xvf ILSVRC2012_img_train.tar -C ./train
tar xvf ILSVRC2012_img_val.tar -C ./val

dir=./train 
for x in `ls $dir/*tar` do     
  filename=`basename $x .tar`     
  mkdir $dir/$filename     
  tar -xvf $x -C $dir/$filename 
done 
rm *.tar
