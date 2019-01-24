TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        main.cpp \
    dense.cpp \
    nalu.cpp \
    neural_net.cpp \
    ranker_utils.cpp \
    simple_attention.cpp \
    sru_rnn.cpp \
    ugrnn.cpp

HEADERS += \
    layer.h \
    dense.h \
    nalu.h \
    neural_net.h \
    ranker_utils.h \
    simple_attention.h \
    sru_rnn.h \
    ugrnn.h

