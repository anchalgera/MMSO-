
function alpha_armijo = armijo(sigma,x,s,f0,f1,gamma,beta)

n = 12;% number of triangles
[x,y] = meshgrid(0:(1/n):1,0:(1/n):1);   %1/n = step size

T = delaunay(x,y);% delaunay is used to get admissible triangulation
triplot(T,x,y,'r')
z = zeros(size(x));%to initialize the point

a = (length(z)-1);

for i = 1:length(z)
z(i,1) = 0.5 - norm(0.5-((i-1)/a));%one side triangle
end

for i = 1:length(z)
z(i,length(z)) = 0.5 - norm(0.5-((i-1)/a));  %another side triangle
end
trimesh(T,x,y,z)
  
%finding the surface area 
A = 0;
for i = 1:length(x)^2
  if i<length(x)^2-2
  a1 = [x(i) y(i) z(i)];
  a2 = [x(i+1) y(i+1) z(i+1)];
  a3 = [x(i+2) y(i+2) z(i+2)];

  else
  a1 = [x(i) y(i) z(i)];
  a2 = [x(i-1) y(i-1) z(i-1)];
  a3 = [x(i-2) y(i-2) z(i-2)];

  A = 0.5*(norm( cross((a1-a2),(a1-a3))));
  A = A+i;
  
  end
end
    
x = rand ;   
% Armijo stepsize rule parameters
  sigma = 0.1;
  beta = 0.5;
  gamma = [1e-3,1e-2];
  g=grad(x);
  k=0; % k = iterations
  q=1; % nf = function eval
  %i = 0; 
  
 while ((i > 0)  &&  (g==0) && (norm(s) <= 1e-8 ))   
    s = -g;    % steepest descent direction
    a = 1;
    newvalue = alpha_armijo(x + a*s);
    q = q+1;
    if (f0(x)-(f0(x)+sigma*s))>=(-gamma*sigma*f1(x)'*s)
         a = a*beta;
      newvalue = alpha_armijo(x + a*s);
      nf = nf+1;
    end
    
 end           
 
function g = grad(x)
g(1) = 100*(2*(x(1)^2-x(2))*2*x(1)) + 2*(x(1)-1);
g(2) = 100*(-2*(x(1)^2-x(2)));
g = g';   
end    

end