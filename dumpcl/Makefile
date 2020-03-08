TARGET := $(notdir $(shell pwd))

all: $(TARGET)

%: %.c
	clang -framework OpenCL $^ -o $@

.phony: all clean

clean:
	$(RM) $(TARGET) 

