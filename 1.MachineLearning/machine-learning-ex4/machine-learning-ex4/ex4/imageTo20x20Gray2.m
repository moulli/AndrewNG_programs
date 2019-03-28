function vectorImage = imageTo20x20Gray(fileName, cropPercentage=0, rotStep=0)
  
  
  Image3DmatrixRGB = imread(fileName);
  Image3DmatrixYIQ = rgb2ntsc(Image3DmatrixRGB );
  Image2DmatrixBW  = Image3DmatrixYIQ(:,:,1);
  
  oldSize = size(Image2DmatrixBW);
  cropDelta = floor((oldSize - min(oldSize)) .* (cropPercentage/100));
  finalSize = oldSize - cropDelta;
  cropOrigin = floor(cropDelta / 2) + 1;
  copySize = cropOrigin + finalSize - 1;
  croppedImage = Image2DmatrixBW( ...
                    cropOrigin(1):copySize(1), cropOrigin(2):copySize(2));
  scale = [20 20] ./ finalSize;
  newSize = max(floor(scale .* finalSize),1)
  
  rowIndex = min(round(((1:newSize(1))-0.5)./scale(1)+0.5), finalSize(1));
  colIndex = min(round(((1:newSize(2))-0.5)./scale(2)+0.5), finalSize(2));
  
  newImage = zeros(length(rowIndex), length(colIndex));
  diffplus = 5; % Le nombre de valeurs qu'on ne va pas prendre dans rowdiff et coldiff
  rowdiff = floor((rowIndex(2)-rowIndex(1))/2) - diffplus;
  coldiff = floor((colIndex(2)-colIndex(1))/2) - diffplus;
  for i = 1:length(rowIndex)
    for j = 1:length(colIndex)
      mprov = croppedImage((rowIndex(i)-rowdiff):(rowIndex(i)+rowdiff), ...
                (colIndex(j)-coldiff):(colIndex(j)+coldiff));
      moy_mprov = sum(mprov(:))/length(mprov(:));
      newImage(i, j) = moy_mprov;
    end
  end
  
  newAlignedImage = rot90(newImage, rotStep);
  invertedImage = - newAlignedImage;
  maxValue = max(invertedImage(:));
  minValue = min(invertedImage(:));
  delta = maxValue - minValue;
  normImage = (invertedImage - minValue) / delta;
  
  % Add contrast. Multiplication factor is contrast control.
  contrastedImage = sigmoid((normImage -0.5) * 10);
  
  imshow(contrastedImage, [-1, 1] );
  vectorImage = reshape(contrastedImage, 1, newSize(1)*newSize(2));

end