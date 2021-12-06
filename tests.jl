###################### Tests, used in the development of the code, but not actually used after
function test_conv() # test on convolution
	n = 50
	X = rand(n)
	R = rand(2n-1)
	vrai = [sum([R[i-j+n]*X[j] for j=1:n]) for i=1:n]
	c = n
	cv = DSP.conv(X,R)
	test = cv[c:c+n-1]
	px(vrai,"\n",test,"\n",cv,"\n Length cv= ",length(cv))
	boo = sum(abs.(test-vrai)) < 1e-4
	px("Equal ? ",boo)
end

function test_herm(R,p) # tests the relation conj(R_{2Nk-m}) = R_m for the output of vε_fourier
	s = sum(abs.([conj(R[2*p.Nk - m]) - R[m] for m=1:p.Nk]))
	if s > 1e-4
		px("R not hermitian, ",s)
	end
	s
end

function test_arr_vec(p)
	u = rand(p.Nk)
	uu = arr2vec(vec2arr(u,p),p)
	uuu = vec2arr(arr2vec(u,p),p)
	px(sum(abs.(uu .- u))," ",sum(abs.(uuu .- u)))
end

function test_arrfft_vec(p)
	u = rand(p.Nk)
	uu = vec2arr_fft(arr_fft2vec(u,p),p)
	uuu = arr_fft2vec(vec2arr_fft(u,p),p)
	@assert sum(abs.(uu .- u)) < 1e-5
	@assert sum(abs.(uuu .- u)) < 1e-5
end

function test_fft()
	n = 10
	u = rand(n)
	uu = FFTViews.fft(FFTViews.ifft(u))
	uuu = FFTViews.ifft(FFTViews.fft(u))
	px(sum(abs.(uu .- u))," ",sum(abs.(uuu .- u)))
end

function test_prod_vec(p) # DOES NOT WORK
	u,v = rand(p.Nk),rand(p.Nk)
	uf = FFTViews.fft(u); vf = FFTViews.fft(v)
	ua = arr_fft2vec(uf,p)
	va = arr_fft2vec(vf,p)
	uv_vec = DSP.conv(ua,va)[p.Nk:2*p.Nk-1]
	uv_a = DSP.conv(uf,vf)[p.Nk:2*p.Nk-1]
	uv_a = DSP.conv(uf,vf)[p.Nk:2*p.Nk-1]
	uv_v = arr2vec(u.*v,p)
	dir_uv_a = FFTViews.fft(u.*v)

	pl = plot([real.(uv_a),real.(dir_uv_a)])
	display(pl)
	px(abs.(uv_vec ./ uv_v))
	px(sum(abs.(uv_vec/m  .- uv_v)))
end

function test_fft(a,p)
	c = vec(FFTViews.fft(a))
	N = length(a)
	cc = [sum([a[n]*exp(-im*2pi*(n-1)*(k-1)/N) for n=1:N]) for k=1:N]
	px("Test FFT ",sum(abs.(c .- cc)))
end

function test_arr_vec(u,p) # verifies it's an involution
	a = vec2arr(u,p)
	v = arr2vec(a,p)
	x = sum(abs.(u.-v))
	aa = arr2vec(u,p)
	vv = vec2arr(aa,p)
	x += sum(abs.(u.-vv))
	px("Test arr vec conversions ",x)
end
