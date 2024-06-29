function lejaOrderedPoints = leja(points)
    % Check if input is empty
    if isempty(points)
        error('Input must be a non-empty vector of points.');
    end
    
    % Start with the point having the smallest magnitude
    [~, idx] = min(abs(points));
    lejaOrderedPoints = points(idx);
    points(idx) = []; % Remove this point from further consideration
    
    while ~isempty(points)
        % Compute distances from remaining points to the last point in lejaOrder
        distances = abs(points - lejaOrderedPoints(end));
        
        % Find the index of the point with the maximum minimum distance
        [~, nextIdx] = max(distances);
        
        % Append this point to the lejaOrderedPoints
        lejaOrderedPoints = [lejaOrderedPoints, points(nextIdx)];
        
        % Remove this point from further consideration
        points(nextIdx) = [];
    end
end


function r = seprts(p)
  % r = seprts(p)
  % This program is for spectral factorization.
  % The roots on the unit circle must have even degree.
  % Roots with high multiplicity will cause problems,
  % they should be handled by extracting them prior to
  % using this program.
  SN = 0.0001; % Small Number (criterion for deciding if a
  % root is on the unit circle).
  rts = roots(p);
  % The roots INSIDE the unit circle
  irts = rts(abs(rts)<(1-SN));
  % The roots ON the unit circle
  orts = rts((abs(rts)>=(1-SN)) & (abs(rts)<=(1+SN)));
  N = length(orts);
  if rem(N,2) == 1
    disp(’Sorry, but there is a problem (1) in seprts.m’)
    r = [];
    return
  end
  % Sort roots on the unit circle by angle
  [a,k] = sort(angle(orts));
  orts = orts(k(1:2:end));
  % Make final list of roots
  r = [irts; orts];
end


function [h,r] = sfact(p)
  % [h,r] = sfact(p)
  % spectral factorization of a polynomial p.
  % h: new polynomial
  % r: roots of new polynomial
  %
  % % example:
  % g = rand(1,10);
  % p = conv(g,g(10:-1:1));
  % h = sfact(p);
  % p - conv(h,h(10:-1:1)) % should be 0
  % Required subprograms: seprts.m, leja.m
  % leja.m is by Markus Lang, and is available from the
  % Rice University DSP webpage: http://www.dsp.rice.edu/
  if length(p) == 1
    h = p;
    r = [];
    return
  end
  % Get the appropriate roots.
  r = seprts(p);
  % Form the polynomial from the roots
  r = leja(r);
  h = poly(r);
  if isreal(p)
    h = real(h);
  end
  % normalize
  h = h*sqrt(max(p)/sum(abs(h).^2));
end
