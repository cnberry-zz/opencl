TARGET := $(notdir $(shell pwd))
  
all: $(TARGET)

%: %.cpp %.cl
	g++ -o $@ $< -lOpenCL

.phony: all clean

clean:
	$(RM) $(TARGET)
