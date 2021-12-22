include("lobpcg.jl")
include("conj_grad.jl")
include("tests.jl")
using Plots, Printf, LinearAlgebra, DSP, FFTViews, Optim, IterativeSolvers, LaTeXStrings#, PyPlot
pyplot()

px = println
float2str(x,n=3) = @sprintf("%.2E", x) # n is the number of digits

function ∇(a,dx) # derivation in periodic boundary conditions
	N = length(a)
	na = [a[i+1]-a[i] for i=1:N-1]
	push!(na,a[1]-a[N])
	na*dx
end

# Parameters
mutable struct Params 
	D; N; ε
	M; Nk # M is the cutoff
	k_list # = [-M,...,M]
	k_list_squared; k_list_squared_inv
	dx; x_axis
	function Params(D,N,M)
		p = new(D,N)
		p.M = M
		p.k_list = [-M + i for i=0:2M]/N
		p.k_list_squared = p.k_list.^2
		p.k_list_squared_inv = 1 ./ max.(p.k_list_squared,1)
		p.Nk = 2M+1
		p.dx = 2pi*N/p.Nk
		p.x_axis = [p.dx*i for i=0:p.Nk-1]
		p
	end
end

####################### Homogenization quantities for v(x,y) = a cos nx cos qy + b cos mx sin py

mutable struct Potential # v(x,y) = a*cos(n*x)*cos(k*y) + b * cos(m*x) *sin(p*y)
	a; b; n; m; k; p; Py; vxy; Vf; χ; χ2; ∇yχ; ∇x∇yχ; ∇1A; η; A; A1; v_integrated; v_int; G_a; vε; χε; χ2ε; ∇yχε; ∇x∇yχε; ∇1Aε; ηε; Aε; A1ε; V; G
	function Potential(a,b,n,m,k,p)
		if k == p
			px("p and q have to be different in this simulation")
			@assert k != p
		end
		new(a,b,n,m,k,p,max(abs(n),abs(m)))
	end
end

coefs(v) = (v.a,v.b,v.n,v.m,v.k,v.p)

function v2pot(pot,v) 
	(a,b,n,m,k,p) = coefs(v)
	f = (pot == "v" ? (x,y) -> a*cos(n*x)*cos(k*y) + b * cos(m*x) *sin(p*y) :
		pot == "v_int" ? x -> a*cos(n*x) + b * cos(m*x) :
		pot == "G" ? x -> -3*((2p == -k || 2p == k) ? 1 : 0)*(a*b^2 /(8*p^4)) *(cos(m*x))^2 *cos(n*x) :
		pot == "χ" ? (x,y) -> -(a/(k^2))*cos(n*x)*cos(k*y) -(b/(p^2))*cos(m*x)*sin(p*y) :
		pot == "γ" ? (x,y) -> (a/(k^4))*cos(n*x)*cos(k*y) +(b/(p^4))*cos(m*x)*sin(p*y) :
		pot == "A" ? (x,y) -> -(a/(k^3))*cos(n*x)*sin(k*y) +(b/(p^3))*cos(m*x)*cos(p*y) :
		pot == "∇1A" ? (x,y) -> (n*a/(k^3))*sin(n*x)*sin(k*y) -(m*b/(p^3))*sin(m*x)*cos(p*y) :
		pot == "∇yχ" ? (x,y) -> (a/k)*cos(n*x)*sin(k*y) -(b/p)*cos(m*x)*cos(p*y) :
		pot == "∇x∇yχ" ? (x,y)-> -(n*a/k)*sin(n*x)*sin(k*y) +(m*b/p)*sin(m*x)*cos(p*y) :
		pot == "η" ? (x,y) -> (a^2/(8*k^4))*(cos(n*x))^2 *(cos(2k*y)) - (b^2/(8*p^4))*(cos(m*x))^2*cos(2p*y) + a*b*0.5*(1/k^2 + 1/p^2) *cos(m*x)*cos(n*x)*(sin((p-k)*y)/((p-k)^2) + sin((k+p)*y)/((p+k)^2)) :
		#pot == "v_eff" ? x -> -0.5*((a/k)^2*(cos(n*x))^2 +(b/p)^2*(cos(m*x))^2))
		x -> -0.5*((a/k)^2*(cos(n*x))^2 +(b/p)^2*(cos(m*x))^2))
	f
end

function generate_funs_ε(v,ε,p)
	v.V = v.Vf.(p.x_axis); v.v_int = v.v_integrated.(p.x_axis)
	v.G = v2pot("G",v); v.G_a = v.G.(p.x_axis)
	v_ep(x) = v.vxy(x,x/ε); v.vε = v_ep.(p.x_axis)
	χ_ep(x) = v.χ(x,x/ε); v.χε = χ_ep.(p.x_axis)
	χ2_ep(x) = v.χ2(x,x/ε); v.χ2ε = χ2_ep.(p.x_axis)
	∇yχ_ep(x) = v.∇yχ(x,x/ε); v.∇yχε = ∇yχ_ep.(p.x_axis)
	∇x∇yχ_ep(x) = v.∇x∇yχ(x,x/ε); v.∇x∇yχε = ∇x∇yχ_ep.(p.x_axis)
	∇1A_ep(x) = v.∇1A(x,x/ε); v.∇1Aε = ∇1A_ep.(p.x_axis)
	η_ep(x) = v.η(x,x/ε); v.ηε = η_ep.(p.x_axis)
	A_ep(x) = v.A(x,x/ε); v.Aε = A_ep.(p.x_axis)
	A1_ep(x) = v.A1(x,x/ε); v.A1ε = A1_ep.(p.x_axis)
end

function generate_funs_effective(v)
	v.vxy = v2pot("v",v); v.v_integrated = v2pot("v_int",v); v.χ = v2pot("χ",v); v.A = v2pot("A",v); v.∇1A = v2pot("∇1A",v); v.∇yχ = v2pot("∇yχ",v); v.∇x∇yχ = v2pot("∇x∇yχ",v); v.η = v2pot("η",v); v.Vf = v2pot("v_eff",v)
	v.χ2 = (x,y) -> v.η(x,y) -2*v.∇1A(x,y)
	v.A1 = (x,y) -> -2*v.A(x,y)
end

function vε_fourier(Dε,Nε,pot,p) # Fourier modes of vε
	N = 2*p.Nk-1; v = zeros(ComplexF64,N)
	for j=1:N
		l = j - p.Nk
		(a,b,n,m,k,P) = coefs(pot)
		pN = Dε*P; kN = Dε*k; nD = Nε*n; mD = Nε*m
		x1 = 0; x2 = 0
		if -l+nD-kN == 0 || -l-nD-kN == 0 || -l-nD+kN == 0 || -l+nD+kN == 0
			x1 = 1
		end
		if -l+mD-pN == 0 || -l-mD-pN == 0
			x2 = 1
		elseif -l+mD+pN == 0 || -l-mD+pN == 0
			x2 = -1
		end
		v[j] = x1*a/4 + x2*im*b/4
	end
	return v # size 2*Nk -1, computes R_n = F_{n-Nk}
end

function Veff_fourier(pot,p) # Fourier modes of the effective potential V_0
	(a,b,n,m,k,P) = coefs(pot)
	nD = p.N*n; mD = p.N*m; N = 2*p.Nk-1
	V = zeros(ComplexF64,N)
	for j=1:N
		l = j - p.Nk
		x1 = 0; x2 = 0; x3 = 0
		if l == 0;                     x1 = 1; end
		if -l-2nD == 0 || -l+2nD == 0; x2 = 1; end
		if -l-2mD == 0 || -l+2mD == 0; x3 = 1; end
		V[j] = (a/k)^2*(2*x1 + x2) + (b/P)^2 * (2*x1 + x3)
	end
	return -(1/8)*V
end

function G_fourier(pot,p) # Fourier modes of the effective potential V_0
	(a,b,n,m,k,P) = coefs(pot)
	nD = p.N*n; mD = p.N*m
	N = 2*p.Nk-1
	V = zeros(ComplexF64,N)
	if 2P != k && 2P != -k
		return V
	end
	for j=1:N
		l = j - p.Nk; x = 0
		if -l+2mD+nD == 0; x += 1; end
		if -l+2mD-nD == 0; x += 1; end
		if -l-2mD+nD == 0; x += 1; end
		if -l-2mD-nD == 0; x += 1; end
		if -l-nD == 0 || -l+nD == 0; x += 2; end
		V[j] = x
	end
	return -(3*a*b^2/(64 * P^4)) *V
end


####################### Operators and their eigenmodes

prod_fun_vec(G,u,n) = DSP.conv(u,G)[n:2n-1] # Gives the product G u in vector components, where G is a potential for instance

M2(u,∇u,ε,v) = (1 .+ ε.*v.χε).*u .+ ε^2 .*(v.χ2ε.*u .+ v.A1ε .* ∇u)

actionK(K,p) = [K^2 + p.k_list[i]^2 + 2*K*p.k_list[i] for i=1:p.Nk]
actionV0(v,p) = X -> prod_fun_vec(Veff_fourier(v,p),X,p.Nk)
actionVε(Dε,Nε,v,p) = X -> (Dε/Nε)*prod_fun_vec(vε_fourier(Dε,Nε,v,p),X,p.Nk)

actionH0(K,v,p) = X -> actionK(K,p).*X + actionV0(v,p)(X)
actionH1(H0,ε,v,p) = X -> H0(X) + ε*prod_fun_vec(G_fourier(v,p),X,p.Nk)
actionHε(Dε,Nε,K,v,p) = X -> actionK(K,p).*X + actionVε(Dε,Nε,v,p)(X)

Kinetic(ϕ,K,p) = real(ϕ'*(actionK(K,p).*ϕ))/p.N # kinetic energy of ϕ, which is in Fourier
PotEnergy_ε(ϕ,v,p) = real(ϕ'*actionVε(p.D,p.N,v,p)(ϕ))/p.N # potential energy with vε/ε

# Finds the first l eigenmodes of the action H
solve_lobpcg_hom(H,l,p;maxiter=100,tol=1e-5) = solve_lobpcg(H,p.Nk,l,p.k_list_squared;maxiter=maxiter,tol=tol)

function eigenmodes(A,Nε,l,p,maxiter,tol,computes=true)
	if computes
		qN = sqrt(Nε)
		(λ,ϕ,conv_ε) = solve_lobpcg_hom(A,l,p;maxiter=maxiter,tol=tol)
		λ = λ[l]*Nε
		u_fourier = qN*ϕ[l]
		u = vec2arr(u_fourier,p)
		∇u = vec2arr(vec2∇vec(u_fourier,p),p)
		return (u_fourier,u,∇u,λ)
	else
		return (fill(1.0,p.Nk),fill(1.0,p.Nk),fill(2.0,p.Nk),1.0)
	end
end

####################### Linear equation

function M2t(u_fourier,z,u_to_compare_four,v,H0,ε,p) # returns (u,M2 tilde u) where... when z is not in the spectrum. u_to_compare_four = U is the one such that M2 u - U is minimized
	Gvec = G_fourier(v,p)
	Gu = prod_fun_vec(Gvec,u_fourier,p.Nk)
	Mperp_fourier = conjugate_gradient(X -> H0(X).-z*X,-Gu,X -> p.k_list_squared_inv.*X,1e-9/p.dx,10000,true)
	Mperp = vec2arr(Mperp_fourier,p)
	u = vec2arr(u_fourier,p)
	U = vec2arr(u_to_compare_four,p)
	∇u = vec2arr(vec2∇vec(u_fourier,p),p)

	(θ,ɑ) = find2gauges(U,M2(u,∇u,ε,v),ε.*(1 .+ε.*v.χε).*Mperp,p,1e-10,1)
	u .*= exp(im*θ); ∇u .*= exp(im*θ)
	Mperp .*= exp(im*ɑ)
	M2u = M2(u,∇u,ε,v) .+ ε.*(1 .+ε.*v.χε).*Mperp
	# M2u = (1 .+ε.*v.χε).*u .+ ε^2 .*((v.ηε .- 2*v.∇1Aε).*u .- 2*v.Aε .* ∇u) .+ ε.*(1 .+ε.*v.χε).*Mperp
	(u,M2u)
end

function solve_linear_problem(Hep,H0,z,f,ε,v,p)
	Uε_vec = conjugate_gradient(X -> Hep(X).-z*X,f,X -> p.k_list_squared_inv.*X,1e-9/p.dx,100,true)
	Uε = vec2arr(Uε_vec,p)
	U0_vec = conjugate_gradient(X -> H0(X).-z*X,f,X -> p.k_list_squared_inv.*X,1e-9/p.dx,100,true)
	(U0,M2U0) = M2t(U0_vec,µ,Uε_vec,v,H0,ε,p) 
	(U0,M2U0,Uε)
end

####################### Manipulations of Fourier and direct functions

# Derivation at the vector level
vec2∇vec(v,p) = im*[v[j]*p.k_list[j] for j=1:p.Nk]

# DFT form to vector
arr_fft2vec(a,p) = vcat(a[p.M+2:2*p.M+1],a[1:p.M+1])*sqrt(2pi*p.N)/length(a)

# vector in the e^{ikx} rep (k=-M to M), to vector â like in the DFT form
vec2arr_fft(v,p) = vcat(v[p.M+1:2*p.M+1],v[1:p.M])*length(v)/sqrt(2pi*p.N) # to arr (a_n 's)
vec2arr(v,p) = FFTViews.ifft(vec2arr_fft(v,p)) # Fourier representation to direct one
arr2vec(a,p) = arr_fft2vec(FFTViews.fft(a),p)


####################### Miscellaneous functions

mean(a) = sum(a)/length(a)
integral(a,p) = sum(a)*2*pi*p.N/length(a) # p.N/length(a) is conserved as N varies
scaprod(u,v,p) = integral(conj.(u).*v,p)

function normSob(a,s,P) # s'th Sobolev norm of a
	c = vec(FFTViews.fft(a))
	N = length(c)
	l = if (N%2 ==0) Int(N/2) else Int((N-1)/2) end
	p = if (N%2 ==0) Int(-1+N/2) else Int((N-1)/2) end
	cst = if (s!=0) 0 else abs(c[1])^2 end
	pos = [Float64(abs(k))^(2s) * abs(c[k+1])^2   for k=1:l]
	neg = [Float64(abs(k))^(2s) * abs(c[N-k+1])^2 for k=1:p]
	sqrt((sum(pos) + sum(neg) + cst)*2pi*P.N/N^2)/(sqrt(P.N)*P.N^s)
end

function singular_function(p) # produces a discontinuous function
	center = floor(p.Nk/2); dNx = floor(p.Nk/6)
	farr = [abs(i-center) < dNx ? 1 : 1/2 for i=1:p.Nk]
	farr = farr/normSob(farr,0,p)^2
	arr2vec(farr,p)
end

function find_gauge(a,b,p,tol=1e-10,sob=0) # finds θ such that b = e^{iθ} a
	 res = Optim.optimize(θ -> normSob(b.-a.*exp(im*θ),sob,p), 0, 2pi; rel_tol=tol)
	 Optim.minimizer(res)
end

function find2gauges(a,b,c,p,tol=1e-10,sob=0) # finds α,θ minimizing a - e^{iα} b - e^{iθ} c
	lower = [0,0]; upper = [2pi,2pi]; x0 = [0.0,0.0]
	f(X) = normSob(a .- b.*exp(im*X[1]).- c.*exp(im*X[2]),sob,p)
	res = Optim.optimize(f,x0,NelderMead())
	m = Optim.minimizer(res)
	(m[1],m[2])
end

get_index(key,keys) = findfirst(isequal(key), keys) # obtains the index (of pm.keys) of the key, corresponding to pm. Returns Nothing if it does not exist
function get_index12(key1,key2,keys1,keys2) # gives the index i such that keys1[i] = key1 and keys2[i] = key2
	for i=1:length(keys1)
		if keys1[i] == key1 && keys2[i] == key2
			return i
		end
	end
	return -1
end

####################### Builds series of relevant values of D and N, where ε=N/D

function ND(Dmax,Nmin) # returns two tables, Dεs and Nεs, where D/N goes from 1 to Nmin/Dmax, and pgcds are taken
	Dεs = []; Nεs = []
	for d=Dmax:-1:Nmin
		g = gcd(d,Dmax)
		push!(Dεs,Dmax/g); push!(Nεs,d/g)
	end
	(Dεs,Nεs)
end

function NDs(Dmaxs,Nmins,delete_last=true) 
	Dεs = []; Nεs = []
	for i=1:length(Dmaxs)
		(De,Ne) = ND(Dmaxs[i],Nmins[i])
		Dεs = vcat(Dεs,De); Nεs = vcat(Nεs,Ne)
	end
	if delete_last # delete_last deletes the last element of the array
		Dεs = Dεs[1:end-1]; Nεs = Nεs[1:end-1]
	end
	(Dεs,Nεs)
end


####################### Parameters for plots

function collect_measures(meas,keys) # for each key, creates a table of values
	n = length(keys)
	dict = Dict(keys[i] => [] for i =1:n)
	for i=1:length(meas)
		for (key, value) in dict
			push!(dict[key],meas[i][key])
		end
	end
	dict
end

mutable struct ParamsMeasures # to manipulate measures that we make, and store their parameters
	n; keys; labels; powers; colors; styles; styles_dot; divide_powers
	function ParamsMeasures(div_pow)
		m = new()
		m.divide_powers = div_pow
		m.keys = ["diff_E0","diff_E1","diff_Et1","diff_E2","diff_ψ0_H0","diff_ψ2_H0","diff_ψ1_H1","diff_ψ2_H1","diff_ψt2_H1","Kε","Vε"]
		m.n = length(m.keys)
		m.powers = [1,2,2,4,1,2,1,2,2,0,0]
		power_strings = m.divide_powers ? [string("ε^{-",m.powers[i],"} ") for i=1:length(m.powers)] : fill(" ",length(m.powers))
		pre_labels = ["|E^ε_{\\ell,k}-E^0_{\\ell,k}|",
			    "|E^ε_{\\ell,k}-E^0_{\\ell,k}-\\varepsilon \\int_{\\Omega} V_1 |ψ^0_{\\ell,k}|^2|",
			    "|E^ε_{\\ell,k}-\\widetilde{E}^{\\varepsilon,(1)}_{\\ell,k}|",
			    "|E^ε_{\\ell,k}-E^{\\varepsilon,(2)}_{\\ell,k}|",
			    "\\left| \\! \\left|ψ^ε_{\\ell,k} - ψ^0_{\\ell,k}\\right| \\! \\right|_{L^2}",
			    "\\left| \\! \\left|ψ^ε_{\\ell,k} - ψ^{\\varepsilon,(2)}_{\\ell,k}\\right| \\! \\right|_{L^2}",
			    "\\left| \\! \\left|ψ^ε_{\\ell,k} - \\widetilde{ψ}^{\\varepsilon,(1)}_{\\ell,k}\\right| \\! \\right|_{H^1}",
			    "\\left| \\! \\left|ψ^ε_{\\ell,k} - ψ^{\\varepsilon,(2)}_{\\ell,k}\\right| \\! \\right|_{H^1}",
			    "\\left| \\! \\left|ψ^ε_{\\ell,k} - \\widetilde{ψ}^{\\varepsilon,(2)}_{\\ell,k}\\right| \\! \\right|_{H^1}",
			    "\\frac{1}{p} \\int_\\Omega |\\nabla \\psi^ε_{\\ell,k}|^2",
			    "\\frac{1}{10} (\\int_\\Omega |\\nabla \\psi^ε_{\\ell,k}|^2 + \\frac{1}{2}\\int_\\Omega \\frac{v_ε}{ε} |\\psi^ε_{\\ell,k}|^2)",
			    ]
		m.labels = [LaTeXString(string("\$ ",power_strings[i],pre_labels[i]," \$")) for i=1:m.n]
		m.colors = [:red,:red,:red,:red,:blue,:blue,:green,:green,:green,:blue,:red]
		m.styles = [:dot,:dash,:dash,:solid,:dash,:solid,:dash,:solid,:dot,:dot,:dot]
		m.styles = [m.powers[i]==1 ? :dot : m.powers[i]==2 ? :dash : :solid for i=1:m.n]
		m.styles_dot = [:dashdot,:dot,:dot,:dot,:dashdot,:dot,:dashdot,:dot,:dot,:dot]
		m
	end
end

####################### Plots

plot2d(f,p) = heatmap(p.x_axis,p.y_axis,f,xlabel="x", ylabel="y") # real 2d function

# To plot curves of functions of space
function plot1d(fs,p,lab=fill("",10))
	h = plot()
	colors = [:blue,:green,:brown,:black,:purple,:pink,:yellow,:cyan]
	Nx = Int(floor(p.Nk/p.N))
	windowed = false
	x_windowed = windowed ? p.x_axis[1:Nx] : p.x_axis
	for i=1:length(fs)
		y_windowed = windowed ? fs[i][1:Nx] : fs[i]
		plot!(x_windowed,y_windowed,linewidth = 0.5,label=lab[i],color=colors[i])
	end
	h
end

function plot1d_windowed(fs,p,a,dx,lab=fill("",10))
	Npoints = 100
	ddx = dx/Npoints
	x_axis_windowed = [a + ddx*i for i=1:Npoints]
	h = plot()
	for i=1:length(fs)
		plot!(x_axis_windowed,fs[i],linewidth = 0.5,label=lab[i])
	end
	h
end

function plot_measures(m,Dεs,Nεs,keys,pm,todo)
	powers = true; logx = false; logy = true; legend_on = :none; xsize = 700; seriestype = :line; ylabel =""; alpha = 1; fontsize = 10; legendfontsize = fontsize
	pl = plot(); εs = Nεs ./ Dεs
	if todo=="full_left"
		powers = false
		logx = false; logy = true
		legend_on = :bottomleft
		xsize = 750
	elseif todo=="full_right"
		powers = false
		logx = true; logy = true
		legend_on = false
		xsize = 350
	end
	if todo=="constants"
		powers = true
		logx = true; logy = true
		legend_on = :outerleft
		xsize = 1000
	end
	if todo=="kinpot"
		powers = false
		logx = true; logy = false
		legend_on = :topright
		xsize = 1000
		seriestype = :scatter
		alpha = 0.5
		legendfontsize = 12
		# annotate!(pl,0.5,5,"Coucou",:red)
		# px("KIN ",m["Kε"][i])
		# scatter!(pl,[p/q],[m["Kε"][i]],alpha=alpha,markerstrokewidth=0,color=:green,label=string("(p,q)=(",p,",",q,")"))
		i_max = argmax(m["Kε"])
		px("imax ",i_max)
		ps = [1, 1, 1,1,2,3, 3,11, 4, 3,57,Int(Nεs[i_max])]
		qs = [4, 3, 2,1,3,4, 5,15,15, 7,115,Int(Dεs[i_max])]
		ud = [1,-1,-1,1,1,1,-1,-1,-1,-1,-1,-1]
		shifts = fill(0,length(ps)) #[0.1,0.1,0.1,0.3]
		for j=1:length(ps)
			(p,q) = (ps[j],qs[j])
			i = get_index12(p,q,Nεs,Dεs)
			text = LaTeXString(string("\$ \\frac{",p,"}{",q,"}\$"))
			if i!=-1
				x = p/q+shifts[j]
				sgn = ud[j]
				fact = 0.25
				y0 = m["Kε"][i]; y = y0 + sgn*0.35*fact
				annotate!(pl,x,y,text,:black,annotationfontsize=1)
				plot!([x,x],[y- sgn*0.13*fact,y0 + sgn*0.05*fact],arrow=true,color=:black,linewidth=1,label="") # arrow
			end
		end
	end
	n = length(keys)
	ft = font(fontsize)
	plot!(pl,legendfontsize=legendfontsize,xaxis = ("ɛ",logx ? :log10 : :none,ft),yaxis=(logy ? :log10 : :none,ft),legend=legend_on,xflip=true,size=(xsize,600))
	for s=1:length(keys)
		k = keys[s]; ki = get_index(keys[s],pm.keys)
		y_ax = m[k]./(powers ? εs.^pm.powers[ki] : 1)
		plot!(pl,εs,y_ax,label=pm.labels[ki],color=pm.colors[ki],linestyle=pm.styles[ki],seriestype =seriestype,alpha=alpha,markerstrokewidth=0)
	end
	savefig(pl,string(todo,".png"))
	# display(pl)
end


####################### Solve for one ε

function one_study(Dε,Nε,v,K,l,pr=false,legend=true)
	qN = sqrt(Nε)
	ε = Nε/Dε
	α = 5
	M = max(Int(floor(α*v.Py*l*Dε)))
	n = floor((2M+1)/Nε); M = Int(floor((n*Nε-1)/2)) # 2M+1 has to be a multiple of N for dx to be invariant under change of N but conservation of ε
	p = Params(Dε,Nε,M)
	generate_funs_ε(v,ε,p)
	mea = Dict() # stores measures
	px("========== ε= ",float2str(ε),"=",Nε,"/",Dε,", M= ",M,", Nk= ",p.Nk)

	H0 = actionH0(K,v,p)
	Hε = actionHε(Dε,Nε,K,v,p)
	H1 = actionH1(H0,ε,v,p)

	if false # Linear equation
		µ = -10
		f = singular_function(p) # produces a discontinuous function
		(U0,M2U0,Uε) = solve_linear_problem(Hε,H0,µ,f,ε,v,p)
	end

	# Eigenmodes
	maxiter = 10000; tol = 1e-8
	(ψ0_fourier,ψ0,∇ψ0,λ0) = eigenmodes(H0,Nε,l,p,maxiter,tol,true)
	(ψε_fourier,ψε,∇ψε,λε) = eigenmodes(Hε,Nε,l,p,maxiter,tol,true)
	(ψ01_fourier,ψ01,∇ψ01,λ01) = eigenmodes(H1,Nε,l,p,maxiter,tol,true)

	λ1 = integral(v.G_a .* abs2.(ψ0),p)
	Kε = Kinetic(ψε_fourier,K,p)
	Vε = PotEnergy_ε(ψε_fourier,v,p)

	# Computes ψ1
	θ = find_gauge(ψ0,ψε,p,1e-9)
	ψ0 .*= exp(im*θ); ∇ψ0 .*= exp(im*θ)
	mea["diff_ψ0_H0"]  = normSob(ψε .- ψ0,0,p)
	um = (1 .+ ε.*v.χε).*ψ0
	ψ1 = um/normSob(um,0,p)

	# Computes E1
	Hε_ψ1 = vec2arr(Hε(arr2vec(ψ1,p)),p)
	Et1 = real(scaprod(ψ1,Hε_ψ1,p))

	# Computes ψ2
	M2ψ01 = M2(ψ01,∇ψ01,ε,v)
	θ = find_gauge(M2ψ01,ψε,p,1e-9)
	ψ01 .*= exp(im*θ); ∇ψ01 .*= exp(im*θ)
	M2ψ01 = M2(ψ01,∇ψ01,ε,v)
	ψ2 = M2ψ01/normSob(M2ψ01,0,p)

	# px("Test kinetic energy ",abs(Kε - normSob(ψε,1,p)^2)) # Test equality of kinetic energy with two methods

	# Computes E2
	Hε_ψ2 = vec2arr(Hε(arr2vec(ψ2,p)),p)
	E2 = real(scaprod(ψ2,Hε_ψ2,p))

	# Computes U1
	Gvec = G_fourier(v,p)
	Gψ0 = prod_fun_vec(Gvec,ψ0_fourier,p.Nk)
	U1_vec = conjugate_gradient(X -> H0(X).- λ0*X,λ1*ψ0_fourier .- Gψ0,X -> p.k_list_squared_inv.*X,1e-9/p.dx,100,true)
	U1 = vec2arr(U1_vec,p)
	U1 .-= scaprod(ψ0,U1,p)*ψ0/(normSob(ψ0,0,p)^2)

	# Computes u2, not right yet
	# (θ,ɑ) = find2gauges(ψε,M2(ψ0,∇ψ0,ε,v),ε.*(1 .+ε.*v.χε).*U1,p,1e-10,1) # not right
	# ψ0 .*= exp(im*θ); ∇ψ0 .*= exp(im*θ); U1 .*= exp(im*ɑ)
	# u2 = M2(ψ0,∇ψ0,ε,v) .+ ε.*(1 .+ε.*v.χε).*U1

	# Makes measures
	mea["diff_E0"] = abs(λε- λ0)/Nε
	mea["diff_E1"] = abs(λε- λ0 - ε*λ1)/Nε
	mea["diff_Et1"] = abs(λε- Et1)/Nε
	mea["diff_E2"] = abs(λε- E2)/Nε
	mea["diff_ψ1_H1"]  = normSob(ψε .- ψ1,1,p)
	mea["diff_ψ2_H0"]  = normSob(ψε .- ψ2,0,p)
	mea["diff_ψ2_H1"]  = normSob(ψε .- ψ2,1,p)
	mea["Kε"] = Kε; mea["Vε"] = Vε
	# mea["diff_ψt2_H1"] = normSob(ψε .- u2,1,p)

	# Plots
	if pr
		fac = 50 # to decrease the potentials for plots
		title = string("εsilon=",float2str(ε,3),", n (exc state)=",l,", K=",float2str(K,3)," , Nk=",p.Nk)
		lab = [L"|\psi^\varepsilon_{\ell,k}|^2",L"|\psi^0_{\ell,k}|^2",L"|\widetilde{\psi}^{\varepsilon,(1)}_{\ell,k}|^2",L"|\psi^{\varepsilon,(2)}_{\ell,k}|^2",string("v_int/",fac),string("V/",fac),string("G/",fac),string("v_ε/",fac)]
		graphs = [abs2.(ψε),abs2.(ψ0),abs2.(ψ1),abs2.(ψ2)]
		if false # add potentials to the plot
			graphs = vcat(graphs,[v.v_int,v.V,v.G_a]/fac)
		end
		px("Plots")
		pl = plot1d(graphs,p,lab)
		#pl = plot1d_windowed(to_plot,p,1,min(100*ε,1+2pi),lab)
		plot!(pl,title="",legend=legend,legendfontsize=10,size=(500,400))
		savefig(pl,string("curveN",Nε,"_D",Dε,".png"))
		# display(pl)
	end
	return mea
end


####################### Solves for many ε and produces plots

function study(todo)
	v = Potential(1,-1,2,1,2,1)
	K = 0.; l = 1
	generate_funs_effective(v)
	constants = false; keys = []
	if todo=="plot_curves"
		one_study(3,1,v,K,l,true,true)
		N = 57
		one_study(2N+1,N,v,K,l,true,false)
	else
		constants = false
		(Dεs,Nεs) = NDs([2*3*5*7],[3*7]) # long computation
		if todo=="full_left"
			# (Dεs,Nεs) = NDs([2*3*5],[3])
			push!(Dεs,100); push!(Nεs,5)
		end
		if todo=="full_right"
			keys = ["diff_E0","diff_E1","diff_E2","diff_ψ0_H0","diff_ψ2_H0","diff_ψ1_H1","diff_ψ2_H1"]
			# (Dεs,Nεs) = NDs([2*3*5],[3])
			nmin = log(Dεs[end]/Nεs[end])/log(10)
			len_others = 20
			Dεs = vcat(Dεs,Int.(floor.(10 .^(range(nmin,stop=3.5,length=len_others))))); Nεs = vcat(Nεs,fill(1,len_others))
		end
		if todo=="constants"
			constants = true
			keys = ["diff_E0","diff_E1","diff_E2","diff_ψ0_H0","diff_ψ2_H0","diff_ψ1_H1","diff_ψ2_H1"]
			# (Dεs,Nεs) = NDs([2*3*5],[3])
			nmin = log(Dεs[end]/Nεs[end])/log(10)
			len_others = 10
			Dεs = vcat(Dεs,Int.(floor.(10 .^(range(nmin,stop=2.1,length=len_others))))); Nεs = vcat(Nεs,fill(1,len_others))
		end
		if todo=="kinpot"
			keys = ["Kε"]
			(Dεs,Nεs) = NDs([2*3*5],[3])
			nmin = log(Dεs[end]/Nεs[end])/log(10)
			len_others = 5
			Dεs = vcat(Dεs,Int.(floor.(10 .^(range(nmin,stop=3,length=len_others))))); Nεs = vcat(Nεs,fill(1,len_others))
			# Dεs = Int.(floor.(10 .^(range(1,stop=3,length=len_others)))); Nεs = fill(1,len_others)
		end
		εs = Nεs ./ Dεs
		px("l= ",l,"\n Dεs = ",Dεs,"\n Nεs = ",Nεs,"\n εs = ",εs)
		pm = ParamsMeasures(constants)
		meas = []
		for i=1:length(Dεs)
			push!(meas,one_study(Dεs[i],Nεs[i],v,K,l))
		end
		meas = collect_measures(meas,keys)
		plot_measures(meas,Dεs,Nεs,keys,pm,todo)
	end
end

# study("constants")
# study("full_left")
# study("full_right")
# study("kinpot")
# study("plot_curves")
