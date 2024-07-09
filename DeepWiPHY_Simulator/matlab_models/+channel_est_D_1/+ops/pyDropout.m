function Y = pyDropout(X, p, train)
%PYDROPOUT Returns dropout of the input X
% at::Tensor at::dropout(const at::Tensor &input, double p, bool train)

import channel_est_D_1.ops.*

Xval = X.value;

p = p.value;
if train
    Scale = 1/(1 - p);
    Mask = rand(size(Xval),'like',Xval) >= p;
    Yval = Scale .* Xval .* Mask;
else
    Yval = Xval;
end

Yrank = X.rank;
Y = struct('value', Yval, 'rank', Yrank);
end