
function gibbsDemoPotts()
pixelX = 100;
pixelY = 100;

Js = [0.5, 1.44, 5];
seeds = [1,2];
Xsamp = zeros(pixelX, pixelY, length(seeds), length(Js));
for t=1:length(Js)
  J = Js(t);
  for trial=1:length(seeds)
    seed  = seeds(trial);
    rand('state', seed);
    figure(1);clf;
    Xsamp(:,:,trial,t) = gibbs(pixelX, pixelY, J);
  end
end

figure(2);clf
for t=1:length(Js)
  for trial=1:length(seeds)
    subplot2(length(seeds),length(Js),trial,t)
    imagesc(Xsamp(:,:,trial,t)); colormap(autumn); axis off
    J = Js(t);
    title(sprintf('J=%3.2f',  J))
    fname = sprintf('gibbsDemoIsing-J%d-trial%d.mat', J*100, trial);
    printPmtkFigure(fname);
  end
end

%%%%%%%%%%%

function X = gibbs(pixelX, pixelY, J)

X = ( 2 .* ( rand( pixelX, pixelY ) > 0.5 ) - 1 );
for iter = 1 : 50000
  if iter>10000 & mod(iter,5000) == 0
    imagesc(X); colormap(autumn); title(sprintf('J=%5.3f iter=%d', J, iter))
    drawnow
  end
  % Select a pixel at random
  ix = ceil( pixelX * rand(1) );  
  iy = ceil( pixelY * rand(1) );
  pos = ( ix - 1 ) * pixelY + iy; % Index of pixel
  % Find indicies of neighbours
  neighborhood = pos + [-1 1 -pixelY pixelY]; 
  % Remove indices outside picture
  neighborhood( find( [iy == 1    iy == pixelY    ix == 1   ix == pixelX] ) ) = []; 
  w1 = J * sum(  X(neighborhood) == 1);  
  w2 = J * sum(  X(neighborhood) == 2);  
  w3 = J * sum(  X(neighborhood) == 3);  
  w4 = J * sum(  X(neighborhood) == 4);  
  w5 = J * sum(  X(neighborhood) == 5);  
  y1 = exp(w1) / (exp(w1) + exp(w2) + exp(w3) + exp(w4) + exp(w5));
  y2 = exp(w2) / (exp(w1) + exp(w2) + exp(w3) + exp(w4) + exp(w5));
  y3 = exp(w3) / (exp(w1) + exp(w2) + exp(w3) + exp(w4) + exp(w5));
  y4 = exp(w4) / (exp(w1) + exp(w2) + exp(w3) + exp(w4) + exp(w5));
  y5 = exp(w5) / (exp(w1) + exp(w2) + exp(w3) + exp(w4) + exp(w5));
  r = rand(1);
  if r < y1;
    X( iy, ix ) = 1;
  elseif r < y1 + y2;
    X( iy, ix ) = 2;
  elseif r < y1 + y2 + y3;
    X( iy, ix ) = 3;
  elseif r < y1 + y2 + y3 + y4;
    X ( iy, ix) = 4;
  else
    X ( iy, ix ) = 5;
  end 
end
