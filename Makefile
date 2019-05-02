# Forward to src/ file
results:
	$(MAKE) -C src results

release:
	$(MAKE) -C src release
errmsg:
	$(MAKE) -C src errmsg
ptx:
	$(MAKE) -C src ptx
profile:
	$(MAKE) -C src profile
res:
	$(MAKE) -C src res
debug:
	$(MAKE) -C src debug
run:
	$(MAKE) -C src run
clean:
	$(MAKE) -C src clean
