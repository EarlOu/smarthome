TARGET	=	hog

CC	=	g++
CFLAGS	=	-g -Wall
CFLAGS	=	-O3
INC	=	`pkg-config --cflags opencv`
LIB	=	`pkg-config --libs opencv`

BINDIR	=	bin
OBJDIR	=	obj
SRCDIR	=	src

SRC	=	$(wildcard $(SRCDIR)/*.cpp)
OBJS	=	$(SRC:$(SRCDIR)/%.cpp=$(OBJDIR)/%.o)

$(BINDIR)/$(TARGET): $(OBJS)
	$(CC) -o $@ $(OBJS) $(CFLAGS) $(LIB)

-include ${OBJS:.o=.d}

$(OBJS): $(OBJDIR)/%.o : $(SRCDIR)/%.cpp
	$(CC) -MMD -MP -c $< -o $@ $(CFLAGS) $(INC) 

clean:
	@rm -rf $(OBJDIR)/*.o; \
	rm -rf $(OBJDIR)/*.d; \
	rm -rf $(BINDIR)/$(TARGET)
