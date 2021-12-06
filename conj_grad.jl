# Conjugate gradient algorithm
# Solves Ha=b. Kaasschieter 88 "Preconditioned conjugate gradients for solving singular systems", pour justification qd L singulier
# Need L positive and symmetric or hermitian. Can have a kernel but then there can be several solutions

function conjugate_gradient(L,b,precon_inv,tol,maxiter=100,print_infos=false) # L has to be hermitian and positive definite !
	N = length(b)
	p,z0,z1,x = zeros(N),zeros(N),zeros(N),zeros(N)
	r0,r1 = b,b
	i = 0
	err = 1e10
	while i <= maxiter && err > tol
		z0 = z1
		z1 = precon_inv(r1)
		zr = z1' * r1
		β = (i==0) ? 0 :  zr/ (z0' * r0)
		p = z1 .+ β * p
		α = zr / (p' * L(p))
		x = x .+ α*p
		err = sum(abs.(r1))
		r0 = r1
		r1 = r1 .- α*L(p)
		i += 1
	end
	if print_infos
		println("CG converged, ",err<tol,", ",i," steps, error ",sum(abs.(L(x) .- b)))
	end
	x
end

function test_cg()
	n= 1000
	M = randn(ComplexF64,n,n)
	M = M*M'
	D = Diagonal([1/M[i,i] for i=1:n])
	b = rand(n)
	A = M \ b
	L = X -> M*X
	precon_inv = X -> D*X
	tol = 1e-3
	(x,i,boo) = conjugate_gradient(L,b,precon_inv,tol,10000)
	println("\n",A,"\n",x,"\n Niter ",i)
	println("Test CG ",sum(abs.(A .- x))/n)
end

#test_cg()
